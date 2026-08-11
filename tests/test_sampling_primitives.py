"""CPU-only tests for public ASMC sampling and cache primitives."""

import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from asmc_batched import reorder_past_key_values
from asmc_sampler import (
    ASMCConfig,
    Particle,
    build_stop_token_ids,
    compute_answer_masses,
    compute_ess_from_logw,
    compute_power_proposal_logprobs,
    weighted_voting_output,
)


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 3
    all_special_ids = [2, 3, 4]


class SamplingPrimitiveTest(unittest.TestCase):
    def test_public_defaults_are_fixed_and_unconstrained(self):
        config = ASMCConfig()
        self.assertFalse(config.enable_adaptive)
        self.assertFalse(config.legacy_stop_constraints)
        self.assertFalse(config.use_source_weight)
        self.assertIsNone(config.c_int_cap)

    def test_adaptive_particle_counts_follow_nominal_n(self):
        config = ASMCConfig(n_particles=16, enable_adaptive=True)
        self.assertEqual(config.fast_n_particles, 8)
        self.assertEqual(config.hard_n_particles, 16)

    def test_invalid_defensive_mixture_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "epsilon"):
            ASMCConfig(epsilon=0.0)

    def test_nonfinite_and_nonpositive_caps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            ASMCConfig(early_stop_mass_threshold=float("nan"))
        with self.assertRaisesRegex(ValueError, "c_int_cap"):
            ASMCConfig(c_int_cap=0)
        with self.assertRaisesRegex(ValueError, "c_int_cap"):
            ASMCConfig(c_int_cap=float("inf"))

    def test_paper_voting_default_does_not_apply_source_multipliers(self):
        particles = [
            Particle([0, 10], 0.0, -1.0, -1.0, True),
            Particle([0, 20], 0.2, -1.0, -1.0, True),
        ]

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return str(token_ids[-1])

        def fake_parse(text, return_source=False):
            parsed = ("A", "boxed") if text == "10" else ("B", "fallback_number")
            return parsed if return_source else parsed[0]

        with patch("asmc_sampler.parse_answer_robust", side_effect=fake_parse):
            masses, _, _ = compute_answer_masses(particles, Tokenizer(), 1)
            answer, _, vote_info = weighted_voting_output(
                particles, Tokenizer(), 1
            )
            legacy_answer, _, _ = weighted_voting_output(
                particles, Tokenizer(), 1, use_source_weight=True
            )

        self.assertGreater(masses["B"], masses["A"])
        self.assertEqual(answer, "B")
        self.assertFalse(vote_info["use_source_weight"])
        self.assertEqual(legacy_answer, "A")

    def test_default_termination_ids_only_include_eos(self):
        self.assertEqual(build_stop_token_ids(_Tokenizer()), [2])
        self.assertEqual(
            build_stop_token_ids(_Tokenizer(), include_all_special=True),
            [2, 3, 4],
        )

    def test_defensive_mixture_is_normalized(self):
        logits = torch.tensor([0.2, -0.7, 1.3, 0.0])
        log_q = compute_power_proposal_logprobs(
            logits, alpha_t=2.5, epsilon=0.05
        )
        self.assertTrue(torch.isfinite(log_q).all())
        self.assertAlmostEqual(float(log_q.exp().sum()), 1.0, places=6)

    def test_uniform_log_weights_have_full_ess(self):
        self.assertAlmostEqual(compute_ess_from_logw([0.0] * 8), 8.0, places=8)

    def test_legacy_cache_reorder_duplicates_ancestors(self):
        key = torch.arange(3 * 2 * 4 * 1).reshape(3, 2, 4, 1)
        value = key + 100
        ancestors = [2, 2, 0]

        reordered = reorder_past_key_values(((key, value),), ancestors)

        expected = torch.tensor(ancestors)
        self.assertTrue(torch.equal(reordered[0][0], key.index_select(0, expected)))
        self.assertTrue(torch.equal(reordered[0][1], value.index_select(0, expected)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_legacy_cache_reorder_handles_offloaded_layers(self):
        cpu_key = torch.arange(3 * 1 * 2).reshape(3, 1, 2, 1)
        cpu_value = cpu_key + 10
        cuda_key = cpu_key.to("cuda")
        cuda_value = cpu_value.to("cuda")
        ancestors = [2, 0, 2]

        reordered = reorder_past_key_values(
            ((cpu_key, cpu_value), (cuda_key, cuda_value)),
            ancestors,
            device=torch.device("cpu"),
        )

        expected_cpu = torch.tensor(ancestors)
        expected_cuda = expected_cpu.to("cuda")
        self.assertTrue(
            torch.equal(
                reordered[0][0], cpu_key.index_select(0, expected_cpu)
            )
        )
        self.assertTrue(
            torch.equal(
                reordered[1][0], cuda_key.index_select(0, expected_cuda)
            )
        )

    def test_tiny_qwen_cache_reorder_matches_full_replay_logits(self):
        """Exercise ancestor duplication/reordering through real Qwen attention."""
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except (ImportError, AttributeError) as error:
            self.skipTest(f"Transformers Qwen2 classes are unavailable: {error}")

        config = Qwen2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260809)
            model = Qwen2ForCausalLM(config).eval()

        # Each row has a distinct history. The ancestor map both reorders rows
        # and duplicates row 2, as happens during ASMC resampling.
        prefixes = torch.tensor(
            [
                [1, 11, 12, 13, 14],
                [1, 21, 22, 23, 24],
                [1, 31, 32, 33, 34],
            ],
            dtype=torch.long,
        )
        ancestor_indices = [2, 2, 0, 1]
        ancestor_tensor = torch.tensor(ancestor_indices, dtype=torch.long)
        next_tokens = torch.tensor([[41], [42], [43], [44]], dtype=torch.long)

        with torch.no_grad():
            prefix_outputs = model(prefixes, use_cache=True)
            reordered_cache = reorder_past_key_values(
                prefix_outputs.past_key_values,
                ancestor_indices,
                device=prefixes.device,
            )
            cached_outputs = model(
                next_tokens,
                past_key_values=reordered_cache,
                use_cache=True,
            )

            replay_inputs = torch.cat(
                [prefixes.index_select(0, ancestor_tensor), next_tokens], dim=1
            )
            replay_outputs = model(replay_inputs, use_cache=False)

        cached_logits = cached_outputs.logits[:, -1, :]
        replay_logits = replay_outputs.logits[:, -1, :]
        self.assertEqual(tuple(cached_logits.shape), (4, config.vocab_size))
        torch.testing.assert_close(
            cached_logits,
            replay_logits,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
