import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_asmc_only.sh"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_all.sh"
BASELINE_FLAGS = {
    "--run_greedy",
    "--run_naive",
    "--run_std",
    "--run_mcmc",
    "--run_majority",
    "--run_bestofn",
}


class ASMCOnlyScriptContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.capture_path = self.temp_path / "captured-args"
        self.python_stub = self._make_stub("python-stub")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_stub(self, name):
        path = self.temp_path / name
        path.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\0' \"$@\" > \"${ASMC_CAPTURE_ARGS}\"\n"
            "exit \"${ASMC_STUB_EXIT:-0}\"\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _base_env(self):
        env = os.environ.copy()
        for name in (
            "SLURM_ARRAY_TASK_ID",
            "ASMC_BATCH_ARRAY",
            "ASMC_CONDA_ENV",
            "ASMC_MODE",
            "ASMC_N_PROBLEMS",
            "ASMC_RUN_PROFILE",
            "ASMC_STUB_EXIT",
        ):
            env.pop(name, None)
        env.update(
            {
                "ASMC_CAPTURE_ARGS": str(self.capture_path),
                "ASMC_PYTHON": str(self.python_stub),
                "ASMC_RESULTS_DIR": str(self.temp_path / "results"),
            }
        )
        return env

    def _run(self, script, *arguments, env=None):
        return subprocess.run(
            ["bash", str(script), *arguments],
            cwd=ROOT,
            env=env or self._base_env(),
            text=True,
            capture_output=True,
            check=False,
        )

    def _captured_args(self):
        data = self.capture_path.read_bytes()
        return [part.decode() for part in data.split(b"\0") if part]

    def test_default_run_is_full_fixed_asmc_only(self):
        result = self._run(RUN_SCRIPT)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self._captured_args()
        self.assertEqual(arguments[0], "asmc_full_comparison.py")
        self.assertIn("--run_asmc", arguments)
        self.assertIn("--fixed", arguments)
        self.assertNotIn("--n_problems", arguments)
        self.assertIn("--n_particles=64", arguments)
        self.assertIn("--max_tokens=3072", arguments)
        self.assertIn("--anneal_tokens=512", arguments)
        self.assertIn("--hard_anneal_tokens=768", arguments)
        self.assertIn("--attn_implementation=flash_attention_2", arguments)
        self.assertTrue(BASELINE_FLAGS.isdisjoint(arguments))

    def test_comparison_cli_keeps_every_baseline_opt_in(self):
        tree = ast.parse(
            (ROOT / "asmc_full_comparison.py").read_text(encoding="utf-8")
        )
        actions = {}
        true_default_overrides = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_defaults"
            ):
                for keyword in node.keywords:
                    flag = f"--{keyword.arg}" if keyword.arg else None
                    if (
                        flag in BASELINE_FLAGS
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        true_default_overrides.add(flag)
            if not node.args:
                continue
            first_argument = node.args[0]
            if not (
                isinstance(first_argument, ast.Constant)
                and first_argument.value in BASELINE_FLAGS
            ):
                continue
            keyword_values = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
            action = keyword_values.get("action")
            actions[first_argument.value] = (
                action.value if isinstance(action, ast.Constant) else None
            )
        self.assertEqual(set(actions), BASELINE_FLAGS)
        self.assertTrue(all(action == "store_true" for action in actions.values()))
        self.assertFalse(true_default_overrides)

    def test_smoke_adaptive_run_forwards_only_whitelisted_parameters(self):
        env = self._base_env()
        env.update(
            {
                "ASMC_RUN_PROFILE": "smoke",
                "ASMC_MODE": "adaptive",
                "ASMC_N_PROBLEMS": "3",
                "ASMC_C_INT_CAP": "123.5",
                "ASMC_MODEL_REVISION": "immutable-revision",
                "ASMC_N_PARTICLES": "7",
                "ASMC_MAX_TOKENS": "100",
                "ASMC_ANNEAL_TOKENS": "10",
                "ASMC_HARD_ANNEAL_TOKENS": "20",
                "ASMC_HARD_ALPHA_START": "1.2",
                "ASMC_HARD_ESS_THRESHOLD": "0.7",
                "ASMC_ATTN_IMPLEMENTATION": "eager",
            }
        )
        result = self._run(RUN_SCRIPT, "2", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self._captured_args()
        self.assertIn("--adaptive", arguments)
        self.assertEqual(arguments[arguments.index("--n_problems") + 1], "3")
        self.assertEqual(arguments[arguments.index("--c_int_cap") + 1], "123.5")
        self.assertEqual(
            arguments[arguments.index("--model_revision") + 1],
            "immutable-revision",
        )
        self.assertIn("--n_particles=7", arguments)
        self.assertIn("--hard_n_particles=7", arguments)
        self.assertIn("--max_tokens=100", arguments)
        self.assertIn("--anneal_tokens=10", arguments)
        self.assertIn("--hard_anneal_tokens=20", arguments)
        self.assertIn("--hard_alpha_start=1.2", arguments)
        self.assertIn("--hard_ess_threshold=0.7", arguments)
        self.assertIn("--attn_implementation=eager", arguments)
        self.assertTrue(BASELINE_FLAGS.isdisjoint(arguments))

    def test_smoke_defaults_to_one_problem(self):
        env = self._base_env()
        env["ASMC_RUN_PROFILE"] = "smoke"
        result = self._run(RUN_SCRIPT, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self._captured_args()
        self.assertEqual(arguments[arguments.index("--n_problems") + 1], "1")
        self.assertIn("--n_particles=4", arguments)
        self.assertIn("--hard_n_particles=4", arguments)
        self.assertIn("--max_tokens=256", arguments)
        self.assertIn("--anneal_tokens=64", arguments)
        self.assertIn("--hard_anneal_tokens=64", arguments)
        self.assertIn("--attn_implementation=sdpa", arguments)

    def test_invalid_batch_and_problem_count_fail_before_python(self):
        result = self._run(RUN_SCRIPT, "5")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture_path.exists())

        env = self._base_env()
        env["ASMC_N_PROBLEMS"] = "101"
        result = self._run(RUN_SCRIPT, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture_path.exists())

    def test_python_failure_status_is_preserved(self):
        env = self._base_env()
        env["ASMC_STUB_EXIT"] = "37"
        result = self._run(RUN_SCRIPT, env=env)
        self.assertEqual(result.returncode, 37)

    def test_submit_defaults_to_full_five_batch_asmc_campaign(self):
        sbatch_stub = self._make_stub("sbatch")
        env = self._base_env()
        env["PATH"] = f"{self.temp_path}:{env['PATH']}"
        result = self._run(SUBMIT_SCRIPT, "--partition=gpu", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self._captured_args()
        self.assertIn("--partition=gpu", arguments)
        self.assertIn("--array=0-4", arguments)
        self.assertEqual(arguments[-1], "scripts/run_asmc_only.sh")
        self.assertNotIn("scripts/run_full_comparison.sh", arguments)
        self.assertTrue(sbatch_stub.exists())

    def test_submit_smoke_and_custom_safe_array(self):
        self._make_stub("sbatch")
        env = self._base_env()
        env["PATH"] = f"{self.temp_path}:{env['PATH']}"
        env["ASMC_RUN_PROFILE"] = "smoke"
        result = self._run(SUBMIT_SCRIPT, "asmc_only", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--array=0", self._captured_args())

        env["ASMC_BATCH_ARRAY"] = "1-3%2"
        result = self._run(SUBMIT_SCRIPT, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--array=1-3%2", self._captured_args())

    def test_submit_rejects_full_mode_and_unsafe_array_overrides(self):
        self._make_stub("sbatch")
        env = self._base_env()
        env["PATH"] = f"{self.temp_path}:{env['PATH']}"
        result = self._run(SUBMIT_SCRIPT, "full", env=env)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture_path.exists())

        env["ASMC_BATCH_ARRAY"] = "0-5"
        result = self._run(SUBMIT_SCRIPT, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture_path.exists())

        env.pop("ASMC_BATCH_ARRAY")
        result = self._run(SUBMIT_SCRIPT, "--array=0-5", env=env)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture_path.exists())

    def test_scripts_pass_bash_syntax_check(self):
        for script in (RUN_SCRIPT, SUBMIT_SCRIPT):
            result = subprocess.run(
                ["bash", "-n", str(script)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
