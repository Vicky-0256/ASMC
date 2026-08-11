"""CPU-only regression tests for single-sample decoding baselines."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from asmc_full_comparison import (  # noqa: E402
    METHOD_PROTOCOLS,
    _completion_status,
    _record_completion_evidence,
    greedy_sample,
    mcmc_power_sample,
    naive_majority_vote,
    naive_temp_sample,
    parse_answer_robust,
    std_sample,
)
from bestofn import resolved_sampling_generation_kwargs  # noqa: E402


class _Model:
    def __init__(self, continuation=(7, 2)):
        self.last_generate_kwargs = None
        self.continuation = list(continuation)

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        prompt = kwargs["input_ids"]
        continuation = torch.tensor(
            [self.continuation], dtype=torch.long
        )
        result = SimpleNamespace(
            sequences=torch.cat((prompt.cpu(), continuation), dim=1)
        )
        if kwargs.get("output_logits"):
            logits = []
            for token_id in continuation[0]:
                step_logits = torch.full((1, 10), -10.0)
                step_logits[0, token_id] = 10.0
                logits.append(step_logits)
            result.logits = tuple(logits)
            result.scores = tuple(logits)
        return result


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 2

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids if token_id != 2]
        return " ".join(str(token_id) for token_id in token_ids)


class BaselineDecodingTest(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.sampler = SimpleNamespace(
            model=self.model,
            tokenizer=_Tokenizer(),
            device=torch.device("cpu"),
        )

    def assert_full_generation_contract(self, kwargs):
        expected = resolved_sampling_generation_kwargs(self.sampler.tokenizer)
        for name, value in expected.items():
            self.assertIn(name, kwargs)
            self.assertEqual(kwargs[name], value)

    def test_low_temperature_baseline_really_samples(self):
        tokens, _ = naive_temp_sample(
            self.sampler, [3, 4], temp=0.25, max_new_tokens=12
        )

        self.assertEqual(tokens, [7, 2])
        kwargs = self.model.last_generate_kwargs
        self.assertIs(kwargs["do_sample"], True)
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["max_new_tokens"], 12)
        self.assertNotIn("output_scores", kwargs)
        self.assert_full_generation_contract(kwargs)

    def test_greedy_baseline_disables_sampling(self):
        tokens, _ = greedy_sample(
            self.sampler, [3, 4], max_new_tokens=12
        )

        self.assertEqual(tokens, [7, 2])
        kwargs = self.model.last_generate_kwargs
        self.assertIs(kwargs["do_sample"], False)
        self.assertNotIn("temperature", kwargs)
        self.assert_full_generation_contract(kwargs)

    def test_standard_sampling_does_not_request_unused_scores(self):
        tokens, _ = std_sample(self.sampler, [3, 4], max_new_tokens=12)
        self.assertEqual(tokens, [7, 2])
        kwargs = self.model.last_generate_kwargs
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertNotIn("output_scores", kwargs)
        self.assert_full_generation_contract(kwargs)

    def test_majority_sampling_uses_full_generation_contract(self):
        completion, answer, info = naive_majority_vote(
            self.sampler,
            [3, 4],
            temp=0.25,
            max_new_tokens=12,
            n_samples=2,
        )
        self.assertEqual(completion, "7")
        self.assertEqual(str(answer), "7")
        self.assertEqual(info["selected_token_ids"], [7, 2])
        self.assert_full_generation_contract(
            self.model.last_generate_kwargs
        )

    def test_mcmc_ignores_prompt_eos_and_decodes_only_completion(self):
        full_tokens, completion, acceptance = mcmc_power_sample(
            self.sampler,
            context=[2, 9],  # Includes both EOS and a prompt-only number.
            temp=0.25,
            mcmc_steps=0,
            max_new_tokens=2,
            block_num=1,
        )

        self.assertEqual(full_tokens, [2, 9, 7, 2])
        self.assertEqual(completion, "7")
        self.assertEqual(acceptance, 0.0)
        kwargs = self.model.last_generate_kwargs
        self.assertNotIn("output_scores", kwargs)
        self.assertIs(kwargs["output_logits"], True)
        self.assert_full_generation_contract(kwargs)

    def test_immediate_eos_is_valid_empty_text_with_lossless_evidence(self):
        self.model = _Model(continuation=(2,))
        self.sampler.model = self.model

        methods = {
            "greedy": greedy_sample(
                self.sampler, [3, 4], max_new_tokens=2
            ),
            "naive": naive_temp_sample(
                self.sampler, [3, 4], temp=0.25, max_new_tokens=2
            ),
            "std": std_sample(
                self.sampler, [3, 4], max_new_tokens=2
            ),
        }
        for method, (token_ids, completion) in methods.items():
            self.assertEqual(token_ids, [2])
            self.assertEqual(completion, "")
            row = {}
            _record_completion_evidence(row, method, token_ids, 2)
            self.assertEqual(row[f"{method}_completion_token_ids"], "[2]")
            self.assertIs(row[f"{method}_completion_has_eos"], True)

        completion, answer, vote_info = naive_majority_vote(
            self.sampler,
            [3, 4],
            temp=0.25,
            max_new_tokens=2,
            n_samples=1,
        )
        self.assertEqual(completion, "")
        self.assertIsNone(answer)
        self.assertEqual(vote_info["selected_token_ids"], [2])

        full_ids, completion, _ = mcmc_power_sample(
            self.sampler,
            context=[3, 4],
            temp=0.25,
            mcmc_steps=0,
            max_new_tokens=2,
            block_num=1,
        )
        self.assertEqual(full_ids, [3, 4, 2])
        self.assertEqual(completion, "")

    def test_protocol_ids_change_with_the_explicit_generation_contract(self):
        self.assertEqual(
            METHOD_PROTOCOLS["greedy"], "deterministic-greedy-decoding-v2"
        )
        self.assertEqual(
            METHOD_PROTOCOLS["naive"], "single-temperature-sample-v2"
        )
        self.assertEqual(
            METHOD_PROTOCOLS["std"], "single-temperature-one-sample-v2"
        )
        self.assertEqual(
            METHOD_PROTOCOLS["mcmc"],
            "completion-only-eos-mcmc-power-sampling-v4",
        )
        self.assertEqual(
            METHOD_PROTOCOLS["majority"],
            "independent-sampling-unweighted-answer-majority-v2",
        )
        self.assertTrue(METHOD_PROTOCOLS["bestofn"].endswith("argmax-v3"))

    def test_runner_robust_parser_accepts_unboxed_final_answers(self):
        self.assertEqual(
            parse_answer_robust("Reasoning here.\nFinal Answer: 42"), "42"
        )
        self.assertEqual(
            parse_answer_robust("Reasoning here.\n17"), "17"
        )

    def test_completed_with_errors_has_nonzero_exit_code(self):
        self.assertEqual(_completion_status({"asmc": 0}), ("complete", 0))
        self.assertEqual(
            _completion_status({"asmc": 1}),
            ("completed_with_errors", 1),
        )


if __name__ == "__main__":
    unittest.main()
