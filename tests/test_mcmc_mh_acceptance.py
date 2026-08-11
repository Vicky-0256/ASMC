"""CPU-only regression tests for MCMC suffix MH acceptance."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from asmc_full_comparison import (  # noqa: E402
    _mcmc_generated_token_log_probs,
    _mcmc_log_acceptance_ratio,
    _mcmc_log_ratio_terms,
    mcmc_power_sample,
)


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 2

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token != 2]
        return " ".join(str(token) for token in token_ids)


class _ScriptedGenerationModel:
    """Return scripted continuations with uniform target/proposal logits."""

    def __init__(self, continuations):
        self.continuations = list(continuations)
        self.calls = []

    def generate(self, input_ids, **kwargs):
        continuation = self.continuations.pop(0)
        self.calls.append(
            {
                "prefix": input_ids[0].tolist(),
                "max_new_tokens": kwargs["max_new_tokens"],
                "kwargs": dict(kwargs),
            }
        )
        if len(continuation) > kwargs["max_new_tokens"]:
            raise AssertionError("scripted continuation exceeds generation cap")

        continuation_tensor = torch.tensor(
            [continuation], dtype=input_ids.dtype, device=input_ids.device
        )
        sequences = torch.cat((input_ids, continuation_tensor), dim=1)
        logits = tuple(
            torch.zeros((1, 16), dtype=torch.float, device=input_ids.device)
            for _ in continuation
        )
        return SimpleNamespace(
            sequences=sequences,
            logits=logits,
        )


def _sampler(continuations):
    return SimpleNamespace(
        model=_ScriptedGenerationModel(continuations),
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
    )


class McmcMhAcceptanceTest(unittest.TestCase):
    def test_proposal_and_power_target_are_rebuilt_from_raw_logits(self):
        logits = torch.tensor(
            [
                [[0.0, 1.0, 2.0]],
                [[2.0, -1.0, 0.5]],
            ]
        )
        tokens = torch.tensor([2, 0])
        proposal, target = _mcmc_generated_token_log_probs(
            logits, tokens, temp=0.25
        )

        expected_proposal = torch.stack(
            [
                torch.log_softmax(logits[0, 0] / 0.25, dim=-1)[2],
                torch.log_softmax(logits[1, 0] / 0.25, dim=-1)[0],
            ]
        )
        expected_target = 4.0 * torch.stack(
            [
                torch.log_softmax(logits[0, 0], dim=-1)[2],
                torch.log_softmax(logits[1, 0], dim=-1)[0],
            ]
        )
        torch.testing.assert_close(torch.tensor(proposal), expected_proposal)
        torch.testing.assert_close(torch.tensor(target), expected_target)

    def test_ratio_terms_include_full_current_tail_and_choice_ratio(self):
        terms = _mcmc_log_ratio_terms(
            current_q_log_probs=[-1.0, -2.0, -3.0],
            current_target_log_probs=[-2.0, -4.0, -6.0],
            proposed_q_log_probs=[-0.5],
            proposed_target_log_probs=[-1.0],
            truncation_offset=1,
            current_n_tokens=3,
            proposed_n_tokens=2,
        )

        # The current target suffix is [-4, -6], not the old clipped [-4].
        self.assertAlmostEqual(terms["target"], -1.0 - (-4.0 - 6.0))
        self.assertAlmostEqual(terms["proposal"], (-2.0 - 3.0) - (-0.5))
        self.assertAlmostEqual(
            terms["truncation_choice"], math.log(3.0 / 2.0)
        )

    def test_unequal_length_move_has_finite_ratio_at_fixed_horizon(self):
        log_ratio = _mcmc_log_acceptance_ratio(
            current_q_log_probs=[-1.0, -2.0, -3.0],
            current_target_log_probs=[-2.0, -4.0, -6.0],
            proposed_q_log_probs=[-0.5],
            proposed_target_log_probs=[-1.0],
            truncation_offset=1,
            current_n_tokens=3,
            proposed_n_tokens=2,
            horizon_n_tokens=3,
        )
        self.assertTrue(math.isfinite(log_ratio))
        expected = (-1.0 - (-4.0 - 6.0))
        expected += (-2.0 - 3.0) - (-0.5)
        expected += math.log(3.0 / 2.0)
        self.assertAlmostEqual(log_ratio, expected)

        reverse_log_ratio = _mcmc_log_acceptance_ratio(
            current_q_log_probs=[-1.0, -0.5],
            current_target_log_probs=[-2.0, -1.0],
            proposed_q_log_probs=[-2.0, -3.0],
            proposed_target_log_probs=[-4.0, -6.0],
            truncation_offset=1,
            current_n_tokens=2,
            proposed_n_tokens=3,
            horizon_n_tokens=3,
        )
        self.assertAlmostEqual(reverse_log_ratio, -log_ratio)

    def test_equal_length_ratio_uses_complete_suffix(self):
        log_ratio = _mcmc_log_acceptance_ratio(
            current_q_log_probs=[-1.0, -2.0, -3.0],
            current_target_log_probs=[-2.0, -4.0, -6.0],
            proposed_q_log_probs=[-0.5, -0.75],
            proposed_target_log_probs=[-1.0, -1.5],
            truncation_offset=1,
            current_n_tokens=3,
            proposed_n_tokens=3,
            horizon_n_tokens=3,
        )
        expected = (-1.0 - 1.5) - (-4.0 - 6.0)
        expected += (-2.0 - 3.0) - (-0.5 - 0.75)
        self.assertAlmostEqual(log_ratio, expected)

    def test_shortened_state_can_lengthen_again_to_fixed_horizon(self):
        sampler = _sampler(
            [
                [4, 5],  # Initial two-token state.
                [2],     # First proposal shortens through EOS.
                [7, 8],  # Next proposal grows to the block horizon again.
            ]
        )
        with patch(
            "asmc_full_comparison.random.randint", return_value=2
        ), patch(
            "asmc_full_comparison.np.random.rand", return_value=0.0
        ) as uniform:
            tokens, completion, acceptance = mcmc_power_sample(
                sampler,
                context=[2, 9],  # Prompt EOS must not terminate completion.
                temp=0.25,
                mcmc_steps=2,
                max_new_tokens=2,
                block_num=1,
            )

        self.assertEqual(tokens, [2, 9, 7, 8])
        self.assertEqual(completion, "7 8")
        self.assertEqual(acceptance, 1.0)
        self.assertEqual(
            [call["max_new_tokens"] for call in sampler.model.calls],
            [2, 2, 2],
        )
        for call in sampler.model.calls:
            self.assertNotIn("output_scores", call["kwargs"])
            self.assertTrue(call["kwargs"]["output_logits"])
            self.assertEqual(call["kwargs"]["top_k"], 0)
            self.assertEqual(call["kwargs"]["top_p"], 1.0)
            self.assertEqual(call["kwargs"]["eos_token_id"], [2])
        uniform.assert_called_once_with()

    def test_equal_length_eos_proposal_can_be_accepted(self):
        sampler = _sampler(
            [
                [4, 5],  # Initial state.
                [7, 2],  # Same length, with generated EOS at the end.
            ]
        )
        with patch("asmc_full_comparison.random.randint", return_value=2):
            tokens, completion, acceptance = mcmc_power_sample(
                sampler,
                context=[2, 9],
                temp=0.25,
                mcmc_steps=1,
                max_new_tokens=2,
                block_num=1,
            )

        self.assertEqual(tokens, [2, 9, 7, 2])
        self.assertEqual(completion, "7")
        self.assertEqual(acceptance, 1.0)

    def test_early_stage_eos_can_grow_at_absolute_final_horizon(self):
        sampler = _sampler(
            [
                [2],          # Stage 1 initialization terminates early.
                [2],          # Stage 1 MH keeps the short EOS state.
                [7, 8, 9, 10],  # Stage 2 MH grows to final horizon T=4.
            ]
        )
        with patch(
            "asmc_full_comparison.random.randint", return_value=2
        ), patch(
            "asmc_full_comparison.np.random.rand", return_value=0.0
        ) as uniform:
            tokens, completion, acceptance = mcmc_power_sample(
                sampler,
                context=[2, 9],
                temp=0.25,
                mcmc_steps=1,
                max_new_tokens=4,
                block_num=2,
            )

        self.assertEqual(tokens, [2, 9, 7, 8, 9, 10])
        self.assertEqual(completion, "7 8 9 10")
        self.assertEqual(acceptance, 1.0)
        # Stage 2 must skip ordinary append from the EOS state and issue its
        # MH proposal directly against the absolute final horizon.
        self.assertEqual(
            [call["max_new_tokens"] for call in sampler.model.calls],
            [2, 2, 4],
        )
        self.assertEqual(
            [call["prefix"] for call in sampler.model.calls],
            [[2, 9], [2, 9], [2, 9]],
        )
        uniform.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
