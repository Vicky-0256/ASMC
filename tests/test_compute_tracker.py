"""CPU-only regression tests for paper compute accounting."""

from pathlib import Path
import sys
import unittest

import torch

ASMC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASMC_ROOT))

from compute_instrumentation import create_wrapped_forward
from compute_tracker import ComputeTracker


class ComputeTrackerTest(unittest.TestCase):
    def test_uniform_prefill_uses_triangular_cost(self):
        tracker = ComputeTracker()
        tracker.log_prefill(batch_size=2, seq_len=3)

        self.assertEqual(tracker.C_int, 2 * (1 + 2 + 3))
        self.assertEqual(tracker.C_tok, 6)
        self.assertEqual(tracker.C_step, 2)
        self.assertEqual(tracker.forward_calls, 1)

    def test_padded_prefill_uses_effective_lengths(self):
        tracker = ComputeTracker()
        tracker.log_prefill(2, 3, effective_lens=torch.tensor([3, 1]))

        self.assertEqual(tracker.C_int, 6 + 1)
        self.assertEqual(tracker.C_tok, 4)

    def test_single_token_cached_decode(self):
        tracker = ComputeTracker()
        tracker.log_decode(batch_size=4, past_len=9)

        self.assertEqual(tracker.C_int, 4 * 10)
        self.assertEqual(tracker.C_tok, 4)
        self.assertEqual(tracker.C_step, 4)
        self.assertEqual(tracker.forward_calls, 1)

    def test_multi_token_cached_decode_sums_query_positions(self):
        tracker = ComputeTracker()
        tracker.log_decode(batch_size=2, past_len=5, step_tokens=3)

        # Per sequence: (5 + 1) + (5 + 2) + (5 + 3) = 21.
        self.assertEqual(tracker.C_int, 42)
        self.assertEqual(tracker.C_tok, 6)

    def test_heterogeneous_cached_decode(self):
        tracker = ComputeTracker()
        tracker.log_decode(
            batch_size=2,
            total_len=8,
            step_tokens=2,
            effective_lens=[8, 6],
        )

        # Effective past lengths are 6 and 4: (7+8) + (5+6) = 26.
        self.assertEqual(tracker.C_int, 26)

    def test_c_step_is_not_forward_call_count(self):
        tracker = ComputeTracker()
        tracker.log_prefill(3, 2)
        tracker.log_decode(2, past_len=2)

        metrics = tracker.get_metrics()
        self.assertEqual(metrics["C_step"], 5)
        self.assertEqual(metrics["forward_calls"], 2)
        self.assertEqual(metrics["n_prefill"], 1)
        self.assertEqual(metrics["n_decode"], 1)

    def test_legacy_result_keys_match_canonical_metrics(self):
        tracker = ComputeTracker()
        tracker.log_prefill(1, 4)
        tracker.log_decode(1, total_len=5)
        stats = tracker.get_stats()

        self.assertEqual(stats["total_flops"], stats["C_int"])
        self.assertEqual(stats["total_tokens"], stats["C_tok"])
        self.assertEqual(stats["prefill_flops"], 10)
        self.assertEqual(stats["decode_flops"], 5)


class ComputeInstrumentationTest(unittest.TestCase):
    @staticmethod
    def _forward(*args, **kwargs):
        return args, kwargs

    def test_wrapper_accounts_for_padding_and_multi_token_decode(self):
        tracker = ComputeTracker()
        wrapped = create_wrapped_forward(
            self._forward, tracker=tracker, sync_cuda=False
        )

        wrapped(
            input_ids=torch.zeros((2, 3), dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )
        # Tuple cache shape: [batch, heads, cached length, head dimension].
        key = torch.zeros((2, 1, 4, 1))
        value = torch.zeros_like(key)
        wrapped(
            input_ids=torch.zeros((2, 2), dtype=torch.long),
            past_key_values=((key, value),),
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]
            ),
        )

        # Prefill: triangle(2) + triangle(3) = 9.
        # Decode: totals 6 and 5 with two query tokens = 11 + 9 = 20.
        self.assertEqual(tracker.C_int, 29)
        self.assertEqual(tracker.C_tok, 5 + 4)
        self.assertEqual(tracker.C_step, 4)
        self.assertEqual(tracker.forward_calls, 2)

    def test_failed_forward_is_not_committed_to_algorithmic_compute(self):
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("CUDA out of memory")
            return args, kwargs

        tracker = ComputeTracker()
        wrapped = create_wrapped_forward(
            fail_once, tracker=tracker, sync_cuda=False
        )
        inputs = torch.zeros((2, 3), dtype=torch.long)

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            wrapped(input_ids=inputs)
        self.assertEqual(tracker.forward_calls, 0)
        self.assertEqual(tracker.C_int, 0)

        wrapped(input_ids=inputs)
        self.assertEqual(tracker.forward_calls, 1)
        self.assertEqual(tracker.C_int, 2 * (1 + 2 + 3))


if __name__ == "__main__":
    unittest.main()
