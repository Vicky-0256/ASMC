import hashlib
import json
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.result_audit import (
    AuditError,
    MATH500_DATASET_SHA256,
    PUBLICATION_METHOD_PROTOCOLS,
    PUBLICATION_RNG_PROTOCOL,
    PUBLICATION_SAMPLING_PROTOCOL_METADATA,
    QWEN25_MATH_7B_OUTPUT_VOCAB_SIZE,
    _load_pinned_math500,
    _publication_sampling_config,
    _verify_publication_ground_truth_row,
    _verify_publication_rng_row,
)


class CompletionTokenEvidenceBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = _load_pinned_math500()[0][3]

    def _row(
        self,
        *,
        method="naive",
        token_ids,
        completion="",
        answer="",
        max_tokens=3072,
    ):
        row = {
            "question": self.canonical["prompt"],
            "correct_answer": self.canonical["answer"],
            "batch_idx": 0,
            f"{method}_answer": answer,
            f"{method}_completion": completion,
            f"{method}_completion_token_ids": json.dumps(
                token_ids, separators=(",", ":")
            ),
            f"{method}_completion_has_eos": 151643 in token_ids,
            "max_tokens": max_tokens,
        }
        return row

    def _verify(self, row, *, method="naive"):
        return _verify_publication_ground_truth_row(
            row,
            method=method,
            problem_idx=3,
            canonical_row=self.canonical,
            location="fixture:2",
        )

    def test_empty_decoded_text_accepts_structurally_valid_nonempty_tokens(self):
        for completion, token_ids, max_tokens in (
            ("", [42, 151643], 3072),
            (" \n\t", [152063, 151643], 3072),
            ("", [42, 43], 2),
        ):
            with self.subTest(completion=completion, token_ids=token_ids):
                recomputed, evidence = self._verify(
                    self._row(
                        token_ids=token_ids,
                        completion=completion,
                        max_tokens=max_tokens,
                    )
                )
                self.assertFalse(recomputed)
                self.assertEqual(evidence, "completion_token_ids")

    def test_rejects_token_ids_outside_pinned_model_output_vocabulary(self):
        self.assertEqual(QWEN25_MATH_7B_OUTPUT_VOCAB_SIZE, 152064)
        for token_id in (152064, 200000):
            with self.subTest(token_id=token_id):
                with self.assertRaisesRegex(AuditError, "output vocabulary range"):
                    self._verify(self._row(token_ids=[token_id, 151643]))

    def test_only_asmc_prefill_budget_exhaustion_allows_empty_token_list(self):
        for max_tokens in (0, 2):
            with self.subTest(max_tokens=max_tokens):
                with self.assertRaisesRegex(
                    AuditError, "allowed only for ASMC"
                ):
                    self._verify(self._row(token_ids=[], max_tokens=max_tokens))

        asmc_row = self._row(method="asmc", token_ids=[])
        asmc_row.update(
            {
                "asmc_budget_exhausted": True,
                "asmc_budget_exhausted_at_token": -1,
                "asmc_stop_reason": "budget_exhausted",
            }
        )
        recomputed, evidence = self._verify(asmc_row, method="asmc")
        self.assertFalse(recomputed)
        self.assertEqual(evidence, "completion_token_ids")

        corrupted = dict(asmc_row, asmc_budget_exhausted_at_token=0)
        with self.assertRaisesRegex(AuditError, "pre-generation budget"):
            self._verify(corrupted, method="asmc")

    def test_eos_only_tokens_cannot_support_nonempty_decoded_text(self):
        row = self._row(token_ids=[151643], completion="Final Answer: 9")
        with self.assertRaisesRegex(AuditError, "EOS-only token artifact"):
            self._verify(row)


class ExactPublicationProtocolAuditTest(unittest.TestCase):
    def test_rejects_near_boundary_flat_sampling_tamper(self):
        row = {
            field: str(value)
            for field, value in PUBLICATION_SAMPLING_PROTOCOL_METADATA.items()
        }
        row["sampling_top_p"] = "0.9999999999995"
        with self.assertRaisesRegex(AuditError, "sampling_top_p"):
            _publication_sampling_config(row, location="fixture:2")

    def test_rejects_old_naive_protocol_even_with_recomputed_rng_key(self):
        old_protocol = "single-temperature-sample-v1"
        common = {
            "attn_implementation": "flash_attention_2",
            "cot": True,
            "dataset_name": "MATH500",
            "dataset_sha256": MATH500_DATASET_SHA256,
            "dtype": "bfloat16",
            "max_new_tokens": 3072,
            "model_id": "Qwen/Qwen2.5-Math-7B",
            "model_revision": "a" * 40,
        }
        identity = {
            "common": common,
            "config": {
                "temperature": 0.25,
                **PUBLICATION_SAMPLING_PROTOCOL_METADATA,
            },
            "protocol": old_protocol,
        }
        rng_key = {
            "base_seed": 7,
            "method": "naive",
            "method_identity": identity,
            "problem_idx": 3,
            "rng_protocol": PUBLICATION_RNG_PROTOCOL,
        }
        payload = json.dumps(
            rng_key,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        row = {
            **common,
            "max_tokens": common["max_new_tokens"],
            "temperature": 0.25,
            "seed": 7,
            "naive_protocol": old_protocol,
            "rng_protocol": PUBLICATION_RNG_PROTOCOL,
            "naive_rng_protocol": PUBLICATION_RNG_PROTOCOL,
            "naive_rng_seed": int(digest[:8], 16),
            "naive_rng_key_sha256": digest,
            "naive_rng_key_payload": payload,
            **{
                field: str(value)
                for field, value in PUBLICATION_SAMPLING_PROTOCOL_METADATA.items()
            },
        }
        with self.assertRaisesRegex(
            AuditError,
            "naive_protocol must be 'single-temperature-sample-v2'",
        ):
            _verify_publication_rng_row(
                row,
                method="naive",
                config="temp0.25",
                problem_idx=3,
                location="fixture:2",
            )

    def test_audit_protocol_map_covers_every_publication_method(self):
        self.assertEqual(
            PUBLICATION_METHOD_PROTOCOLS,
            {
                "greedy": "deterministic-greedy-decoding-v2",
                "asmc": "cache-coherent-asmc-corrected-v1",
                "naive": "single-temperature-sample-v2",
                "std": "single-temperature-one-sample-v2",
                "mcmc": "completion-only-eos-mcmc-power-sampling-v4",
                "majority": (
                    "independent-sampling-unweighted-answer-majority-v2"
                ),
                "bestofn": (
                    "independent-generation-unconditional-length-normalized-"
                    "logprob-argmax-v3"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
