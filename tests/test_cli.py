"""CPU-only tests for experiment-runner CLI safety defaults."""

from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asmc_full_comparison import DEFAULT_MODEL_REVISIONS, resolve_model_revision


RUNNER = Path(__file__).resolve().parents[1] / "asmc_full_comparison.py"


class RunnerCliTest(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_is_importable_without_loading_a_model(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--adaptive", result.stdout)
        self.assertIn("--fixed", result.stdout)
        self.assertIn("--no_asmc", result.stdout)
        self.assertIn("--legacy_stop_constraints", result.stdout)
        self.assertIn("--sequential", result.stdout)
        self.assertIn("--run_bestofn", result.stdout)
        self.assertIn("--bestofn_chunk_size", result.stdout)
        self.assertIn("--c_int_cap", result.stdout)
        self.assertIn("--cint_cap", result.stdout)
        self.assertIn("--hard_anneal_tokens", result.stdout)
        self.assertIn("--hard_alpha_start", result.stdout)
        self.assertIn("--hard_ess_threshold", result.stdout)
        self.assertIn("--no_cot", result.stdout)
        self.assertIn("weighted_no_source matches the", result.stdout)

    def test_asmc_runner_does_not_import_optional_bestofn_eagerly(self):
        probe = (
            "import sys; import asmc_full_comparison; "
            "print(int('bestofn' in sys.modules or 'ASMC.bestofn' in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=RUNNER.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "0")

    def test_qwen_math_release_default_is_an_immutable_revision(self):
        model_id = "Qwen/Qwen2.5-Math-7B"
        revision = resolve_model_revision(model_id, None)
        self.assertEqual(revision, DEFAULT_MODEL_REVISIONS[model_id])
        self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(
            resolve_model_revision(model_id, "caller-selected-revision"),
            "caller-selected-revision",
        )
        self.assertIsNone(resolve_model_revision("another/model", None))

    def test_adaptive_and_fixed_are_mutually_exclusive(self):
        result = self._run("--adaptive", "--fixed")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)

    def test_run_and_no_run_are_mutually_exclusive(self):
        result = self._run("--run_asmc", "--no_asmc")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)

    def test_batched_and_sequential_are_mutually_exclusive(self):
        result = self._run("--batched", "--sequential")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)

    def test_bestofn_candidate_count_must_be_positive(self):
        result = self._run("--bestofn_n", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--bestofn_n must be positive", result.stderr)

    def test_cot_flags_are_mutually_exclusive(self):
        result = self._run("--cot", "--no_cot")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)

    def test_c_int_cap_must_be_positive(self):
        result = self._run("--cint_cap", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--c_int_cap must be positive", result.stderr)

    def test_c_int_cap_accepts_positive_decimal_values(self):
        # The later validation error proves argparse accepted 1.5 as a float
        # without proceeding to model loading.
        result = self._run(
            "--c_int_cap", "1.5", "--n_problems", "0"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--n_problems must be positive", result.stderr)
        self.assertNotIn("invalid int value", result.stderr)

    def test_c_int_cap_must_be_finite(self):
        result = self._run("--c_int_cap", "nan")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--c_int_cap must be finite", result.stderr)

    def test_c_int_cap_requires_batched_backend(self):
        result = self._run("--c_int_cap", "10", "--sequential")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--c_int_cap requires", result.stderr)

    def test_float_arguments_must_be_finite(self):
        result = self._run("--temp", "nan")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--temp must be finite", result.stderr)

    def test_adaptive_hard_pass_arguments_are_validated(self):
        cases = (
            (("--hard_anneal_tokens", "-1"), "must be non-negative"),
            (("--hard_alpha_start", "0"), "must be positive"),
            (("--hard_alpha_start", "nan"), "must be finite"),
            (("--hard_ess_threshold", "1.1"), "must be between 0 and 1"),
            (("--hard_ess_threshold", "nan"), "must be finite"),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                result = self._run(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_mcmc_blocks_must_match_recorded_execution(self):
        result = self._run(
            "--run_mcmc",
            "--mcmc_blocks",
            "5",
            "--max_tokens",
            "3072",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--max_tokens must be divisible by --mcmc_blocks", result.stderr
        )


if __name__ == "__main__":
    unittest.main()
