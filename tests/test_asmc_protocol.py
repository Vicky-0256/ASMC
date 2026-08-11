"""CPU-only regression tests for paper-aligned ASMC runtime semantics."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from asmc_batched import BatchedASMCSampler, batched_asmc_sample
from asmc_sampler import ASMCConfig, ASMCSampler, Particle, asmc_sample
from compute_instrumentation import instrument_model
from compute_tracker import ComputeTracker


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def decode(self, token_ids, skip_special_tokens=True):
        return "unparseable completion"


class _TinyCausalLM(torch.nn.Module):
    """Minimal causal LM supporting sequential and legacy-cache calls."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=128)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, 4), -100.0)
        logits[..., 1] = 100.0

        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
        total_len = past_len + seq_len
        key = torch.zeros(batch_size, 1, total_len, 1)
        value = torch.zeros_like(key)
        return SimpleNamespace(
            logits=logits,
            past_key_values=((key, value),),
        )


class ASMCProtocolTest(unittest.TestCase):
    def test_sequential_backend_rejects_partial_population_caps(self):
        model = _TinyCausalLM()
        tracker = ComputeTracker()
        sampler = ASMCSampler(model, _Tokenizer(), torch.device("cpu"))
        config = ASMCConfig(
            n_particles=1,
            max_new_tokens=5,
            block_size=5,
            c_int_cap=2,
        )

        with self.assertRaisesRegex(ValueError, "batched backend"):
            with instrument_model(model, tracker, sync_cuda=False):
                sampler.sample([3], config, tracker=tracker)

        self.assertEqual(tracker.forward_calls, 0)

    def test_batched_adaptive_passes_share_one_cumulative_tracker(self):
        model = _TinyCausalLM()
        tracker = ComputeTracker()
        sampler = BatchedASMCSampler(model, _Tokenizer(), torch.device("cpu"))
        config = ASMCConfig(
            n_particles=2,
            fast_n_particles=1,
            hard_n_particles=2,
            enable_adaptive=True,
            max_new_tokens=1,
            block_size=1,
            c_int_cap=2,
            use_source_weight=False,
        )

        with patch("asmc_sampler.parse_answer_robust", return_value=(None, "none")):
            with instrument_model(model, tracker, sync_cuda=False):
                _, _, _, diagnostics = sampler.sample(
                    [3], config, tracker=tracker
                )

        # Fast prefill costs 1; hard prefill costs 2 on the same tracker.
        self.assertEqual(tracker.C_int, 3)
        self.assertEqual(tracker.forward_calls, 2)
        self.assertEqual(diagnostics["pass_type"], "hard")
        self.assertTrue(diagnostics["budget_exhausted"])
        self.assertFalse(diagnostics["vote_info"]["use_source_weight"])
        self.assertFalse(
            diagnostics["fast_diag"]["vote_info"]["use_source_weight"]
        )

    def test_final_token_resampling_does_not_advance_an_unused_cache(self):
        model = _TinyCausalLM()
        tracker = ComputeTracker()
        config = ASMCConfig(
            n_particles=2,
            max_new_tokens=1,
            block_size=1,
            ess_threshold=1.0,
            use_source_weight=False,
        )

        # Force the end-of-block resampling branch.  The generated particles
        # are already final at max_new_tokens=1, so there must not be a second
        # model call solely to advance a cache that will never be consumed.
        with patch(
            "asmc_batched.compute_ess_from_logw", return_value=0.0
        ), patch(
            "asmc_batched.compute_answer_masses", return_value=({}, 0, {})
        ):
            with instrument_model(model, tracker, sync_cuda=False):
                _, diagnostics = batched_asmc_sample(
                    model,
                    _Tokenizer(),
                    [3],
                    config,
                    torch.device("cpu"),
                    tracker=tracker,
                )

        self.assertEqual(diagnostics["n_resamples"], 1)
        self.assertEqual(diagnostics["stop_reason"], "max_len")
        self.assertEqual(tracker.forward_calls, 1)

    def test_resampling_checks_early_stop_before_advancing_cache(self):
        for cap in (None, 3):
            with self.subTest(c_int_cap=cap):
                model = _TinyCausalLM()
                tracker = ComputeTracker()
                config = ASMCConfig(
                    n_particles=2,
                    max_new_tokens=2,
                    block_size=1,
                    ess_threshold=1.0,
                    early_stop_min_tokens=0,
                    early_stop_ess_frac=0.0,
                    early_stop_min_parsed_frac=0.0,
                    early_stop_mass_threshold=0.8,
                    early_stop_stable_checks=1,
                    c_int_cap=cap,
                    use_source_weight=False,
                )

                with patch(
                    "asmc_batched.compute_ess_from_logw",
                    side_effect=(0.0, 2.0),
                ), patch(
                    "asmc_batched.compute_answer_masses",
                    return_value=({"42": 1.0}, 2, {}),
                ):
                    with instrument_model(model, tracker, sync_cuda=False):
                        _, diagnostics = batched_asmc_sample(
                            model,
                            _Tokenizer(),
                            [3],
                            config,
                            torch.device("cpu"),
                            tracker=tracker,
                        )

                self.assertEqual(diagnostics["n_resamples"], 1)
                self.assertEqual(diagnostics["stop_reason"], "early_stop")
                self.assertTrue(diagnostics["early_stopped"])
                self.assertFalse(diagnostics["budget_exhausted"])
                self.assertEqual(tracker.forward_calls, 1)

    def test_early_stop_min_tokens_counts_generated_tokens(self):
        config = ASMCConfig(
            n_particles=1,
            max_new_tokens=2,
            block_size=1,
            ess_threshold=0.0,
            early_stop_min_tokens=1,
            early_stop_ess_frac=0.0,
            early_stop_min_parsed_frac=0.0,
            early_stop_mass_threshold=0.8,
            early_stop_stable_checks=1,
            use_source_weight=False,
        )

        for module_name, sample in (
            ("asmc_sampler", lambda model: asmc_sample(
                model, _Tokenizer(), [3], config, torch.device("cpu")
            )),
            ("asmc_batched", lambda model: batched_asmc_sample(
                model, _Tokenizer(), [3], config, torch.device("cpu")
            )),
        ):
            with self.subTest(backend=module_name), patch(
                f"{module_name}.compute_answer_masses",
                return_value=({"42": 1.0}, 1, {}),
            ):
                _, diagnostics = sample(_TinyCausalLM())

            self.assertEqual(diagnostics["tokens_generated"], 1)
            self.assertEqual(diagnostics["early_stop_token"], 0)
            self.assertEqual(diagnostics["stop_reason"], "early_stop")

    def test_cap_consumes_paid_logits_consistently_after_resampling(self):
        observed = []
        for force_resample in (False, True):
            with self.subTest(force_resample=force_resample):
                model = _TinyCausalLM()
                tracker = ComputeTracker()
                config = ASMCConfig(
                    n_particles=2,
                    max_new_tokens=3,
                    block_size=1,
                    ess_threshold=1.0 if force_resample else 0.0,
                    early_stop_min_tokens=99,
                    c_int_cap=3,
                    use_source_weight=False,
                )
                ess_patch = (
                    patch(
                        "asmc_batched.compute_ess_from_logw",
                        side_effect=(0.0, 2.0),
                    )
                    if force_resample
                    else patch(
                        "asmc_batched.compute_ess_from_logw",
                        return_value=2.0,
                    )
                )

                with ess_patch, patch(
                    "asmc_batched.compute_answer_masses",
                    return_value=({}, 0, {}),
                ):
                    with instrument_model(model, tracker, sync_cuda=False):
                        _, diagnostics = batched_asmc_sample(
                            model,
                            _Tokenizer(),
                            [3],
                            config,
                            torch.device("cpu"),
                            tracker=tracker,
                        )

                observed.append(
                    (
                        diagnostics["tokens_generated"],
                        diagnostics["stop_reason"],
                        tracker.forward_calls,
                        tracker.C_int,
                    )
                )

        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[0][0], 2)
        self.assertEqual(observed[0][1], "budget_exhausted")
        self.assertEqual(observed[0][2], 2)

    def test_low_pre_resampling_ess_blocks_early_stop(self):
        config = ASMCConfig(
            n_particles=2,
            max_new_tokens=1,
            block_size=1,
            ess_threshold=1.0,
            early_stop_min_tokens=1,
            early_stop_ess_frac=0.75,
            early_stop_min_parsed_frac=0.0,
            early_stop_mass_threshold=0.8,
            early_stop_stable_checks=1,
            use_source_weight=False,
        )

        for module_name, sample in (
            ("asmc_sampler", lambda model: asmc_sample(
                model, _Tokenizer(), [3], config, torch.device("cpu")
            )),
            ("asmc_batched", lambda model: batched_asmc_sample(
                model, _Tokenizer(), [3], config, torch.device("cpu")
            )),
        ):
            with self.subTest(backend=module_name), patch(
                f"{module_name}.compute_ess_from_logw",
                side_effect=(0.0, 2.0),
            ), patch(
                f"{module_name}.compute_answer_masses",
                return_value=({"42": 1.0}, 2, {}),
            ):
                _, diagnostics = sample(_TinyCausalLM())

            self.assertEqual(diagnostics["n_resamples"], 1)
            self.assertFalse(diagnostics["early_stopped"])
            self.assertEqual(diagnostics["stop_reason"], "max_len")

    def test_early_stop_mass_uses_configured_legacy_source_weighting(self):
        model = _TinyCausalLM()
        config = ASMCConfig(
            n_particles=2,
            max_new_tokens=1,
            block_size=1,
            use_source_weight=True,
        )

        with patch(
            "asmc_batched.compute_answer_masses",
            return_value=({}, 0, {}),
        ) as compute_masses:
            batched_asmc_sample(
                model,
                _Tokenizer(),
                [3],
                config,
                torch.device("cpu"),
            )

        self.assertTrue(compute_masses.call_args.kwargs["use_source_weight"])

    def test_sequential_adaptive_hard_pass_inherits_rejuvenation_setting(self):
        model = _TinyCausalLM()
        sampler = ASMCSampler(model, _Tokenizer(), torch.device("cpu"))
        config = ASMCConfig(
            n_particles=2,
            enable_adaptive=True,
            enable_rejuvenation=False,
        )
        observed_configs = []
        particle = Particle([3, 1], 0.0, 0.0, 0.0, False)

        def fake_sample(model, tokenizer, context, child_config, device, verbose=False, tracker=None):
            observed_configs.append(child_config)
            return [particle], {"budget_exhausted": False}

        def fake_vote(*args, **kwargs):
            return "answer", particle, {"best_mass": 0.0, "n_parsed": 1}

        with patch("asmc_sampler.asmc_sample", side_effect=fake_sample), patch(
            "asmc_sampler.weighted_voting_output", side_effect=fake_vote
        ):
            sampler.sample([3], config)

        self.assertEqual(len(observed_configs), 2)
        self.assertFalse(observed_configs[1].enable_rejuvenation)
        self.assertFalse(observed_configs[1].use_source_weight)


if __name__ == "__main__":
    unittest.main()
