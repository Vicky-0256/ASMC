import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.result_audit import (
    ASMC_HARD_EPSILON,
    AuditError,
    MATH500_DATASET_SHA256,
    PUBLICATION_FULL_SUPPORT_GENERATION_KWARGS,
    PUBLICATION_METHOD_PROTOCOLS,
    PUBLICATION_RNG_PROTOCOL,
    PUBLICATION_SAMPLING_PROTOCOL_METADATA,
    _load_pinned_math500,
    _publication_sampling_config,
    _validate_asmc_config_semantics,
    _validate_asmc_hard_epsilon,
    _verify_publication_ground_truth_row,
    _verify_publication_rng_row,
    audit_csvs,
)
from analysis.select_compute_matched import PUBLICATION_BASELINE_PROTOCOLS
from asmc_full_comparison import METHOD_PROTOCOLS
from asmc_sampler import ASMCConfig
from bestofn import (
    FULL_SUPPORT_SAMPLING_KWARGS,
    resolved_sampling_protocol_metadata,
)


class PinnedGroundTruthAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark_rows, cls.benchmark_artifact = _load_pinned_math500()

    def test_repository_math500_is_content_pinned(self):
        self.assertEqual(len(self.benchmark_rows), 500)
        self.assertEqual(
            self.benchmark_artifact,
            {
                "path": "data/MATH500.json",
                "sha256": MATH500_DATASET_SHA256,
                "n_rows": 500,
            },
        )
        self.assertFalse(Path(self.benchmark_artifact["path"]).is_absolute())

    def test_stdlib_audit_sampling_contract_matches_runtime(self):
        class QwenTokenizerPin:
            eos_token_id = 151643
            pad_token_id = 151643

        self.assertEqual(
            PUBLICATION_FULL_SUPPORT_GENERATION_KWARGS,
            FULL_SUPPORT_SAMPLING_KWARGS,
        )
        self.assertEqual(
            PUBLICATION_SAMPLING_PROTOCOL_METADATA,
            resolved_sampling_protocol_metadata(QwenTokenizerPin()),
        )
        self.assertEqual(
            PUBLICATION_BASELINE_PROTOCOLS,
            {
                method: METHOD_PROTOCOLS[method]
                for method in PUBLICATION_BASELINE_PROTOCOLS
            },
        )
        self.assertEqual(PUBLICATION_METHOD_PROTOCOLS, METHOD_PROTOCOLS)

        row = {
            field: str(value)
            for field, value in PUBLICATION_SAMPLING_PROTOCOL_METADATA.items()
        }
        self.assertEqual(
            _publication_sampling_config(row, location="fixture:2"),
            PUBLICATION_SAMPLING_PROTOCOL_METADATA,
        )
        for field, value in (
            ("sampling_bad_words_ids", "[1]"),
            ("sampling_eos_token_ids", "[1]"),
            ("sampling_policy_payload", "{}"),
            ("sampling_policy_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                corrupted = dict(row)
                corrupted[field] = value
                with self.assertRaisesRegex(AuditError, field):
                    _publication_sampling_config(
                        corrupted, location="fixture:2"
                    )

    def _valid_ground_truth_row(self):
        canonical = self.benchmark_rows[3]
        return canonical, {
            "question": canonical["prompt"],
            "correct_answer": canonical["answer"],
            "batch_idx": 0,
            "asmc_answer": canonical["answer"],
            "asmc_completion": r"Reasoning complete. \boxed{9}",
            "asmc_completion_token_ids": "[42,151643]",
            "asmc_completion_has_eos": True,
            "max_tokens": 3072,
        }

    def test_recomputes_correctness_from_completion_and_cross_checks_answer(self):
        canonical, row = self._valid_ground_truth_row()
        recomputed, evidence = _verify_publication_ground_truth_row(
            row,
            method="asmc",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertTrue(recomputed)
        self.assertEqual(evidence, "completion")

        no_answer = dict(row)
        no_answer.pop("asmc_answer")
        recomputed, evidence = _verify_publication_ground_truth_row(
            no_answer,
            method="asmc",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertTrue(recomputed)
        self.assertEqual(evidence, "completion")

        unparsable = dict(no_answer, asmc_completion="no extractable result")
        recomputed, evidence = _verify_publication_ground_truth_row(
            unparsable,
            method="asmc",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertFalse(recomputed)
        self.assertEqual(evidence, "completion")

        # Corrected publication grading is method-independent and uses the
        # robust parser, not the legacy boxed-only baseline parser.
        for method in (
            "asmc",
            "greedy",
            "naive",
            "std",
            "mcmc",
            "majority",
            "bestofn",
        ):
            with self.subTest(robust_parser_method=method):
                method_row = {
                    "question": canonical["prompt"],
                    "correct_answer": canonical["answer"],
                    "batch_idx": 0,
                    f"{method}_answer": canonical["answer"],
                    f"{method}_completion": "Final Answer: 9",
                    f"{method}_completion_token_ids": "[42,151643]",
                    f"{method}_completion_has_eos": True,
                    "max_tokens": 3072,
                }
                recomputed, evidence = _verify_publication_ground_truth_row(
                    method_row,
                    method=method,
                    problem_idx=3,
                    canonical_row=canonical,
                    location="fixture:2",
                )
                self.assertTrue(recomputed)
                self.assertEqual(evidence, "completion")

        naive_row = {
            "question": canonical["prompt"],
            "correct_answer": canonical["answer"],
            "batch_idx": 0,
            "naive_answer": canonical["answer"],
            "naive_completion": "Final Answer: 9",
            "naive_completion_token_ids": "[42,151643]",
            "naive_completion_has_eos": True,
            "max_tokens": 3072,
        }
        recomputed, evidence = _verify_publication_ground_truth_row(
            naive_row,
            method="naive",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertTrue(recomputed)
        self.assertEqual(evidence, "completion")

        max_length_row = dict(
            naive_row,
            naive_completion_token_ids="[42,43]",
            naive_completion_has_eos=False,
            max_tokens=2,
        )
        recomputed, _ = _verify_publication_ground_truth_row(
            max_length_row,
            method="naive",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertTrue(recomputed)
        too_short = dict(max_length_row, naive_completion_token_ids="[42]")
        with self.assertRaisesRegex(AuditError, "exactly max_tokens"):
            _verify_publication_ground_truth_row(
                too_short,
                method="naive",
                problem_idx=3,
                canonical_row=canonical,
                location="fixture:2",
            )

        identical_wrong_answer = dict(
            row,
            asmc_answer="__definitely_incorrect__",
            asmc_completion=r"\boxed{__definitely_incorrect__}",
        )
        recomputed, evidence = _verify_publication_ground_truth_row(
            identical_wrong_answer,
            method="asmc",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertFalse(recomputed)
        self.assertEqual(evidence, "completion")

    def test_rejects_self_reported_dataset_mapping_and_answer(self):
        canonical, valid = self._valid_ground_truth_row()
        corruptions = (
            ("question", "different question", "question does not match"),
            ("correct_answer", "8", "correct_answer does not match"),
            ("batch_idx", 1, "batch_idx=1 is inconsistent"),
            ("dataset_id", "wrong/id.json", "dataset_id does not match"),
            ("asmc_answer", "8", "asmc_answer is inconsistent"),
        )
        for field, value, message in corruptions:
            with self.subTest(field=field):
                row = dict(valid)
                row[field] = value
                with self.assertRaisesRegex(AuditError, message):
                    _verify_publication_ground_truth_row(
                        row,
                        method="asmc",
                        problem_idx=3,
                        canonical_row=canonical,
                        location="fixture:2",
                    )

    def test_strict_correctness_requires_raw_completion(self):
        canonical, valid = self._valid_ground_truth_row()
        for completion in (None, "", "ERROR: generation failed"):
            with self.subTest(completion=completion):
                row = dict(valid)
                if completion is None:
                    row.pop("asmc_completion")
                else:
                    row["asmc_completion"] = completion
                with self.assertRaisesRegex(AuditError, "completion"):
                    _verify_publication_ground_truth_row(
                        row,
                        method="asmc",
                        problem_idx=3,
                        canonical_row=canonical,
                        location="fixture:2",
                    )

    def test_immediate_eos_is_a_valid_empty_incorrect_completion(self):
        canonical, valid = self._valid_ground_truth_row()
        immediate_eos = dict(
            valid,
            asmc_answer="",
            asmc_completion="",
            asmc_completion_token_ids="[151643]",
        )
        recomputed, evidence = _verify_publication_ground_truth_row(
            immediate_eos,
            method="asmc",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertFalse(recomputed)
        self.assertEqual(evidence, "completion_token_ids")

        corruptions = (
            ("asmc_answer", "9", "cannot have a populated"),
            ("asmc_completion", "visible", "EOS-only token artifact"),
            (
                "asmc_completion_token_ids",
                "[151643,42]",
                "no token after its first EOS",
            ),
            (
                "asmc_completion_token_ids",
                "[42, 151643]",
                "canonical compact JSON",
            ),
            ("asmc_completion_has_eos", False, "does not match"),
        )
        for field, value, message in corruptions:
            with self.subTest(field=field, value=value):
                row = dict(immediate_eos)
                row[field] = value
                with self.assertRaisesRegex(AuditError, message):
                    _verify_publication_ground_truth_row(
                        row,
                        method="asmc",
                        problem_idx=3,
                        canonical_row=canonical,
                        location="fixture:2",
                    )

        missing_has_eos = dict(immediate_eos)
        missing_has_eos.pop("asmc_completion_has_eos")
        with self.assertRaisesRegex(AuditError, "completion_has_eos"):
            _verify_publication_ground_truth_row(
                missing_has_eos,
                method="asmc",
                problem_idx=3,
                canonical_row=canonical,
                location="fixture:2",
            )

        pre_generation_budget = dict(
            valid,
            asmc_answer="",
            asmc_completion="",
            asmc_completion_token_ids="[]",
            asmc_completion_has_eos=False,
            asmc_budget_exhausted=True,
            asmc_budget_exhausted_at_token=-1,
            asmc_stop_reason="budget_exhausted",
        )
        recomputed, evidence = _verify_publication_ground_truth_row(
            pre_generation_budget,
            method="asmc",
            problem_idx=3,
            canonical_row=canonical,
            location="fixture:2",
        )
        self.assertFalse(recomputed)
        self.assertEqual(evidence, "completion_token_ids")

        wrong_empty_budget = dict(pre_generation_budget)
        wrong_empty_budget["asmc_budget_exhausted_at_token"] = 0
        with self.assertRaisesRegex(AuditError, "pre-generation budget"):
            _verify_publication_ground_truth_row(
                wrong_empty_budget,
                method="asmc",
                problem_idx=3,
                canonical_row=canonical,
                location="fixture:2",
            )

    def test_strict_audit_rejects_claimed_correctness_before_averaging(self):
        canonical = self.benchmark_rows[3]
        row = {
            "problem_idx": 3,
            "batch_idx": 0,
            "question": canonical["prompt"],
            "correct_answer": canonical["answer"],
            "asmc_answer": canonical["answer"],
            "asmc_completion": r"Reasoning complete. \boxed{9}",
            "asmc_completion_token_ids": "[42,151643]",
            "asmc_completion_has_eos": True,
            "max_tokens": 3072,
            "asmc_correct": False,
            "asmc_time_s": 1,
            "asmc_c_int": 10,
            "asmc_pass_type": "single",
            "asmc_prefill_flops": 4,
            "asmc_decode_flops": 6,
            "asmc_total_flops": 10,
            "asmc_c_tok": 3,
            "asmc_c_step": 2,
            "asmc_n_forward": 1,
            "asmc_budget_exhausted": False,
            "asmc_budget_exhausted_at_token": "",
            "asmc_stop_reason": "max_len",
            "rng_protocol": "",
            "asmc_rng_protocol": "",
            "asmc_rng_seed": "",
            "asmc_rng_key_sha256": "",
            "asmc_rng_key_payload": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claimed.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(
                AuditError, "does not match repository-recomputed correctness"
            ):
                audit_csvs(
                    [path],
                    method="asmc",
                    config="fixture",
                    run_mode="fixed",
                    expected_problem_count=500,
                    require_provenance=True,
                )


class RngProvenanceAuditTest(unittest.TestCase):
    @staticmethod
    def _valid_rng_row():
        protocol_payload_object = {
            "backend": "batched",
            "config": {"fixture": True},
            "cot": True,
            "vote_mode": "weighted_no_source",
        }
        protocol_payload = json.dumps(
            protocol_payload_object, sort_keys=True, separators=(",", ":")
        )
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
        method_identity = {
            "common": common,
            "config_id": "asmc-fixture",
            "protocol": "cache-coherent-asmc-corrected-v1",
            "protocol_payload": protocol_payload_object,
        }
        key = {
            "base_seed": 7,
            "method": "asmc",
            "method_identity": method_identity,
            "problem_idx": 123,
            "rng_protocol": PUBLICATION_RNG_PROTOCOL,
        }
        payload = json.dumps(
            key,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return {
            **common,
            "max_tokens": common["max_new_tokens"],
            "seed": 7,
            "asmc_protocol": method_identity["protocol"],
            "asmc_protocol_payload": protocol_payload,
            "rng_protocol": PUBLICATION_RNG_PROTOCOL,
            "asmc_rng_protocol": PUBLICATION_RNG_PROTOCOL,
            "asmc_rng_seed": int(digest[:8], 16),
            "asmc_rng_key_sha256": digest,
            "asmc_rng_key_payload": payload,
        }

    def test_recomputes_canonical_method_rng_key(self):
        row = self._valid_rng_row()
        _verify_publication_rng_row(
            row,
            method="asmc",
            config="asmc-fixture",
            problem_idx=123,
            location="fixture:2",
        )

    def test_rejects_tampered_rng_provenance(self):
        corruptions = (
            ("rng_protocol", "other", "rng_protocol must be"),
            ("asmc_rng_protocol", "other", "asmc_rng_protocol must be"),
            ("asmc_rng_seed", 0, "asmc_rng_seed does not match"),
            ("asmc_rng_key_sha256", "0" * 64, "does not match"),
            ("asmc_rng_key_payload", "{}", "canonical RNG key"),
        )
        for field, value, message in corruptions:
            with self.subTest(field=field):
                row = self._valid_rng_row()
                row[field] = value
                with self.assertRaisesRegex(AuditError, message):
                    _verify_publication_rng_row(
                        row,
                        method="asmc",
                        config="asmc-fixture",
                        problem_idx=123,
                        location="fixture:2",
                    )

        with self.assertRaisesRegex(AuditError, "canonical RNG key"):
            _verify_publication_rng_row(
                self._valid_rng_row(),
                method="asmc",
                config="different-config",
                problem_idx=123,
                location="fixture:2",
            )
        with self.assertRaisesRegex(AuditError, "canonical RNG key"):
            _verify_publication_rng_row(
                self._valid_rng_row(),
                method="asmc",
                config="asmc-fixture",
                problem_idx=124,
                location="fixture:2",
            )


class AsmcProtocolSemanticAuditTest(unittest.TestCase):
    def setUp(self):
        self.valid_config = copy.deepcopy(
            vars(ASMCConfig(stop_token_ids=[151643]))
        )

    def test_accepts_runner_config_and_hard_epsilon(self):
        _validate_asmc_config_semantics(self.valid_config)
        _validate_asmc_hard_epsilon(ASMC_HARD_EPSILON)

    def test_rejects_wrong_eos_and_invalid_config_ranges(self):
        corruptions = (
            ("stop_token_ids", [151645], "EOS IDs"),
            ("n_particles", 0, "n_particles must be positive"),
            ("block_size", 1.5, "block_size must be an integer"),
            ("anneal_tokens", -1, "anneal_tokens must be non-negative"),
            ("ess_threshold", 1.1, "ess_threshold must be between"),
            ("epsilon", 0.0, "epsilon must be strictly between"),
            ("alpha_star", 4, "canonical JSON float"),
            ("alpha_star", float("inf"), "alpha_star must be finite"),
            ("eos_penalty", -1.0, "eos_penalty must be non-negative"),
            ("eos_penalty", -0.0, "must not use negative zero"),
            ("use_source_weight", 0, "use_source_weight must be boolean"),
            ("c_int_cap", 4, "canonical JSON float"),
            ("c_int_cap", float("nan"), "c_int_cap must be finite"),
        )
        for field, value, message in corruptions:
            with self.subTest(field=field):
                config = copy.deepcopy(self.valid_config)
                config[field] = value
                with self.assertRaisesRegex(AuditError, message):
                    _validate_asmc_config_semantics(config)

        invalid_counts = copy.deepcopy(self.valid_config)
        invalid_counts["fast_n_particles"] = 5
        invalid_counts["hard_n_particles"] = 4
        with self.assertRaisesRegex(AuditError, "adaptive particle counts"):
            _validate_asmc_config_semantics(invalid_counts)

    def test_hard_epsilon_is_fixed_by_runner_protocol(self):
        for value in (0.07, float("nan"), "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(AuditError, "asmc_hard_epsilon"):
                    _validate_asmc_hard_epsilon(value)


if __name__ == "__main__":
    unittest.main()
