"""CPU-only checks for the cache-resampling microbenchmark CLI."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from microbench import cache_resampling  # noqa: E402


SCRIPT = Path(cache_resampling.__file__).resolve()


class CacheMicrobenchmarkCliTest(unittest.TestCase):
    def _run(self, *args):
        with tempfile.TemporaryDirectory() as temporary_directory:
            return subprocess.run(
                [sys.executable, str(SCRIPT), *args],
                cwd=temporary_directory,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_help_does_not_load_torch_or_a_model(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--model", result.stdout)
        self.assertIn("--revision", result.stdout)
        self.assertIn("--dtype", result.stdout)
        self.assertIn("--device", result.stdout)
        self.assertIn("--N", result.stdout)
        self.assertIn("--L", result.stdout)
        self.assertIn("--output-json", result.stdout)
        self.assertIn("--output-csv", result.stdout)
        self.assertIn("prefix replay", result.stdout)
        self.assertIn("KV-cache ancestor gather", result.stdout)

    def test_particle_count_must_be_positive(self):
        result = self._run("--N", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("greater than zero", result.stderr)

    def test_prefix_length_must_be_positive(self):
        result = self._run("--L", "-1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("greater than zero", result.stderr)

    def test_warmup_may_not_be_negative(self):
        result = self._run("--warmup", "-1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("zero or greater", result.stderr)

    def test_device_must_name_cuda(self):
        result = self._run("--device", "cpu")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a CUDA device", result.stderr)

    def test_output_paths_must_be_distinct(self):
        result = self._run(
            "--output-json",
            "same.out",
            "--output-csv",
            "same.out",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be different paths", result.stderr)


if __name__ == "__main__":
    unittest.main()
