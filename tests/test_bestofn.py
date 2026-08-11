"""CPU-only tests for the public batched Best-of-N baseline."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import unittest

import torch

ASMC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASMC_ROOT))

from bestofn import (
    Candidate,
    FULL_SUPPORT_SAMPLING_KWARGS,
    SAMPLING_POLICY,
    generate_candidates,
    next_smaller_chunk_size,
    normalize_sequence_log_probability,
    resolved_sampling_generation_kwargs,
    resolved_sampling_protocol_metadata,
    sample_best_of_n,
    score_candidates,
    select_best_candidate,
)


def _candidate(*tokens: int) -> Candidate:
    return Candidate(
        tokens=torch.tensor(tokens, dtype=torch.long),
        text=" ".join(str(token) for token in tokens),
        has_eos=False,
        hit_limit=False,
    )


class _OOMScoringModel(torch.nn.Module):
    """CPU fake that simulates a CUDA OOM above a configured batch size."""

    def __init__(self, maximum_batch_size: int = 2):
        super().__init__()
        self.maximum_batch_size = maximum_batch_size
        self.attempted_batch_sizes = []

    def forward(self, input_ids, **kwargs):
        batch_size, sequence_length = input_ids.shape
        self.attempted_batch_sizes.append(batch_size)
        if batch_size > self.maximum_batch_size:
            raise RuntimeError("CUDA out of memory (simulated on CPU)")
        return SimpleNamespace(
            logits=torch.zeros(batch_size, sequence_length, 16)
        )


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def decode(tokens, skip_special_tokens=True):
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {0, 2}]
        return " ".join(str(token) for token in tokens)


class _EndToEndModel(torch.nn.Module):
    """Tiny deterministic stand-in for generation and teacher forcing."""

    def __init__(self):
        super().__init__()
        self.next_token = 3
        self.generation_batch_sizes = []

    def forward(self, input_ids, **kwargs):
        del kwargs
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, 16)
        # Prefer larger candidate token IDs during teacher-forcing scoring.
        logits[..., :] = torch.arange(16, dtype=torch.float)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        batch_size = input_ids.shape[0]
        self.generation_batch_sizes.append(batch_size)
        # Hugging Face generate calls forward internally; this call lets the
        # instrumentation test exercise that same path without a real model.
        self(input_ids=input_ids)
        generated_tokens = torch.arange(
            self.next_token,
            self.next_token + batch_size,
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(1)
        self.next_token += batch_size
        return SimpleNamespace(
            sequences=torch.cat([input_ids, generated_tokens], dim=1)
        )


class _StaticSequenceModel:
    """Return one fixed, rectangular batch of generated sequences."""

    def __init__(self, suffixes):
        self.suffixes = torch.tensor(suffixes, dtype=torch.long)
        self.last_generate_kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        self.last_generate_kwargs = kwargs
        if input_ids.shape[0] != self.suffixes.shape[0]:
            raise AssertionError("test model expects generation in one batch")
        suffixes = self.suffixes.to(
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return SimpleNamespace(sequences=torch.cat([input_ids, suffixes], dim=1))


class _TerminalEOSModel(torch.nn.Module):
    """Generate one visible token and EOS, with deterministic scoring logits."""

    def forward(self, input_ids, **kwargs):
        del kwargs
        batch_size, sequence_length = input_ids.shape
        logits = torch.arange(16, dtype=torch.float).view(1, 1, 16)
        return SimpleNamespace(logits=logits.repeat(batch_size, sequence_length, 1))

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        # Exercise the instrumented generation-forward path before returning a
        # complete generated sequence that terminates with EOS.
        self(input_ids=input_ids)
        suffix = torch.tensor(
            [[3, 2]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).repeat(input_ids.shape[0], 1)
        return SimpleNamespace(sequences=torch.cat([input_ids, suffix], dim=1))


class _ImmediateEOSModel(torch.nn.Module):
    """Generate only terminal EOS while still supporting teacher forcing."""

    def forward(self, input_ids, **kwargs):
        del kwargs
        batch_size, sequence_length = input_ids.shape
        return SimpleNamespace(
            logits=torch.zeros(batch_size, sequence_length, 16)
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        self(input_ids=input_ids)
        eos = torch.full(
            (input_ids.shape[0], 1),
            2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return SimpleNamespace(sequences=torch.cat([input_ids, eos], dim=1))


class BestOfNPrimitiveTest(unittest.TestCase):
    def test_generation_retains_terminal_eos_but_skips_it_when_decoding(self):
        model = _StaticSequenceModel(
            [
                [2, 0, 0],  # immediate EOS followed by padding
                [5, 2, 0],  # one visible token followed by EOS and padding
                [6, 7, 8],  # max-token truncation without EOS
            ]
        )

        candidates, _ = generate_candidates(
            model,
            _Tokenizer(),
            torch.tensor([1, 1]),
            n=3,
            max_new_tokens=3,
            temperature=0.25,
            chunk_size=3,
        )

        self.assertEqual(candidates[0].tokens.tolist(), [2])
        self.assertEqual(candidates[0].text, "")
        self.assertTrue(candidates[0].has_eos)
        self.assertFalse(candidates[0].hit_limit)

        self.assertEqual(candidates[1].tokens.tolist(), [5, 2])
        self.assertEqual(candidates[1].text, "5")
        self.assertTrue(candidates[1].has_eos)
        self.assertFalse(candidates[1].hit_limit)

        self.assertEqual(candidates[2].tokens.tolist(), [6, 7, 8])
        self.assertEqual(candidates[2].text, "6 7 8")
        self.assertFalse(candidates[2].has_eos)
        self.assertTrue(candidates[2].hit_limit)
        expected_policy = resolved_sampling_generation_kwargs(_Tokenizer())
        for name, value in expected_policy.items():
            self.assertEqual(model.last_generate_kwargs[name], value)

    def test_resolved_policy_metadata_is_canonical_and_lossless(self):
        metadata = resolved_sampling_protocol_metadata(_Tokenizer())
        self.assertEqual(metadata["sampling_policy"], SAMPLING_POLICY)
        self.assertEqual(metadata["sampling_min_p"], "none")
        self.assertEqual(metadata["sampling_stop_strings"], "none")
        self.assertEqual(metadata["sampling_eos_token_ids"], "[2]")
        self.assertEqual(metadata["sampling_pad_token_id"], 0)

        payload = metadata["sampling_policy_payload"]
        resolved = json.loads(payload)
        self.assertEqual(resolved["eos_token_id"], [2])
        self.assertEqual(resolved["pad_token_id"], 0)
        self.assertIsNone(resolved["min_p"])
        self.assertIsNone(resolved["stop_strings"])
        self.assertEqual(
            metadata["sampling_policy_sha256"],
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )
        for name, value in FULL_SUPPORT_SAMPLING_KWARGS.items():
            self.assertEqual(resolved[name], value)

        fallback = resolved_sampling_protocol_metadata(
            SimpleNamespace(eos_token_id=2, pad_token_id=None)
        )
        self.assertEqual(fallback["sampling_pad_token_id"], 2)

    def test_immediate_eos_is_a_valid_empty_bestofn_completion(self):
        result = sample_best_of_n(
            _ImmediateEOSModel(),
            _Tokenizer(),
            torch.tensor([1, 1]),
            n=1,
            max_new_tokens=2,
            temperature=0.25,
            generation_batch_size=1,
            scoring_batch_size=1,
            sync_cuda=False,
        )

        self.assertEqual(result.completion, "")
        self.assertEqual(result.candidates[0].tokens.tolist(), [2])
        self.assertTrue(result.candidates[0].has_eos)

    def test_scoring_includes_eos_log_probability_and_length(self):
        model = _TerminalEOSModel()
        candidates = [
            Candidate(
                tokens=torch.tensor([2]),
                text="",
                has_eos=True,
                hit_limit=False,
            ),
            Candidate(
                tokens=torch.tensor([3, 2]),
                text="3",
                has_eos=True,
                hit_limit=False,
            ),
            _candidate(4, 5),
        ]

        scores, _ = score_candidates(
            model,
            torch.tensor([1, 1]),
            candidates,
            chunk_size=3,
            pad_token_id=0,
        )

        token_log_probs = torch.log_softmax(torch.arange(16, dtype=torch.float), dim=0)
        self.assertTrue(torch.isfinite(torch.tensor(scores[0])))
        self.assertAlmostEqual(scores[0], float(token_log_probs[2]), places=6)
        self.assertAlmostEqual(
            scores[1],
            float((token_log_probs[3] + token_log_probs[2]) / 2),
            places=6,
        )
        self.assertAlmostEqual(
            scores[2],
            float((token_log_probs[4] + token_log_probs[5]) / 2),
            places=6,
        )

    def test_terminal_eos_is_included_in_scoring_compute_accounting(self):
        model = _TerminalEOSModel()
        result = sample_best_of_n(
            model,
            _Tokenizer(),
            torch.tensor([1, 1]),
            n=1,
            max_new_tokens=2,
            temperature=0.25,
            generation_batch_size=1,
            scoring_batch_size=1,
            sync_cuda=False,
        )

        self.assertEqual(result.candidates[0].tokens.tolist(), [3, 2])
        self.assertEqual(result.completion, "3")
        self.assertIsNotNone(result.compute)
        # Generation prefill length 2 plus teacher-forcing length 4.  The
        # latter includes the terminal EOS token in the model input.
        self.assertEqual(result.compute["C_tok"], 6)
        self.assertEqual(result.compute["C_int"], 13)
        self.assertEqual(result.compute["forward_calls"], 2)

    def test_length_normalization_changes_the_selected_candidate(self):
        total_log_probabilities = [-0.4, -0.6]
        lengths = [1, 2]

        unnormalized = [
            normalize_sequence_log_probability(score, length, enabled=False)
            for score, length in zip(total_log_probabilities, lengths)
        ]
        normalized = [
            normalize_sequence_log_probability(score, length, enabled=True)
            for score, length in zip(total_log_probabilities, lengths)
        ]

        self.assertEqual(select_best_candidate(unnormalized), 0)
        self.assertEqual(select_best_candidate(normalized), 1)
        self.assertAlmostEqual(normalized[1], -0.3)

    def test_chunk_reduction_is_a_pure_halving_schedule(self):
        chunk_sizes = [9]
        while chunk_sizes[-1] > 1:
            chunk_sizes.append(next_smaller_chunk_size(chunk_sizes[-1]))
        self.assertEqual(chunk_sizes, [9, 4, 2, 1])
        with self.assertRaisesRegex(ValueError, "positive"):
            next_smaller_chunk_size(0)
        with self.assertRaisesRegex(ValueError, "positive"):
            next_smaller_chunk_size(True)

    def test_generation_rejects_nonfinite_or_boolean_parameters(self):
        model = _EndToEndModel()
        prompt = torch.tensor([1, 1])
        for temperature in (float("nan"), float("inf"), True):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(
                ValueError, "temperature must be positive"
            ):
                generate_candidates(
                    model,
                    _Tokenizer(),
                    prompt,
                    n=1,
                    max_new_tokens=1,
                    temperature=temperature,
                )
        with self.assertRaisesRegex(ValueError, "n must be positive"):
            generate_candidates(
                model,
                _Tokenizer(),
                prompt,
                n=True,
                max_new_tokens=1,
                temperature=0.25,
            )

    def test_teacher_forcing_retries_the_failed_batch_at_a_smaller_size(self):
        model = _OOMScoringModel(maximum_batch_size=2)
        candidates = [_candidate(3 + index) for index in range(5)]

        scores, stats = score_candidates(
            model,
            torch.tensor([1, 2]),
            candidates,
            chunk_size=4,
            pad_token_id=0,
        )

        self.assertEqual(model.attempted_batch_sizes, [4, 2, 2, 1])
        self.assertEqual(stats.initial_chunk_size, 4)
        self.assertEqual(stats.final_chunk_size, 2)
        self.assertEqual(stats.oom_retries, 1)
        self.assertTrue(stats.used_oom_fallback)
        self.assertTrue(all(torch.isfinite(torch.tensor(scores))))

    def test_end_to_end_batches_generation_and_tracks_all_forward_calls(self):
        model = _EndToEndModel()
        result = sample_best_of_n(
            model,
            _Tokenizer(),
            torch.tensor([1, 1]),
            n=5,
            max_new_tokens=1,
            temperature=0.25,
            generation_batch_size=2,
            scoring_batch_size=2,
            sync_cuda=False,
        )

        self.assertEqual(model.generation_batch_sizes, [2, 2, 1])
        self.assertEqual(result.best_index, 4)
        self.assertEqual(result.completion, "7")
        self.assertIsNotNone(result.compute)
        # Three generation forwards plus three teacher-forcing forwards.
        self.assertEqual(result.compute["forward_calls"], 6)
        self.assertGreater(result.compute["C_int"], 0)


if __name__ == "__main__":
    unittest.main()
