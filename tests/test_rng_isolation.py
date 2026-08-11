"""CPU-only tests for deterministic per-problem/per-method RNG streams."""

import hashlib
import json
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from asmc_full_comparison import (  # noqa: E402
    RNG_PROTOCOL,
    _activate_method_rng,
    _derive_method_rng_metadata,
    _seed_all_rngs,
)


def _draw_from_all_cpu_rngs():
    return (
        [random.random() for _ in range(3)],
        np.random.random(3).tolist(),
        torch.rand(3).tolist(),
    )


class MethodRngIsolationTest(unittest.TestCase):
    def test_metadata_is_canonical_and_content_addressed(self):
        identity_a = {
            "protocol": "target-v1",
            "config": {"temperature": 0.25, "n": 4},
        }
        identity_b = {
            "config": {"n": 4, "temperature": 0.25},
            "protocol": "target-v1",
        }

        first = _derive_method_rng_metadata(7, 123, "target", identity_a)
        second = _derive_method_rng_metadata(7, 123, "target", identity_b)

        self.assertEqual(first, second)
        self.assertEqual(first["protocol"], RNG_PROTOCOL)
        expected_digest = hashlib.sha256(
            first["key_payload"].encode("utf-8")
        ).hexdigest()
        self.assertEqual(first["key_sha256"], expected_digest)
        self.assertEqual(first["seed"], int(expected_digest[:8], 16))
        self.assertGreaterEqual(first["seed"], 0)
        self.assertLess(first["seed"], 2**32)

        payload = json.loads(first["key_payload"])
        self.assertEqual(payload["base_seed"], 7)
        self.assertEqual(payload["problem_idx"], 123)
        self.assertEqual(payload["method"], "target")
        self.assertEqual(payload["method_identity"], identity_a)
        self.assertEqual(payload["rng_protocol"], RNG_PROTOCOL)

    def test_each_seed_axis_changes_the_stream_key(self):
        identity = {"protocol": "target-v1", "config": {"n": 4}}
        reference = _derive_method_rng_metadata(7, 123, "target", identity)
        variants = [
            _derive_method_rng_metadata(8, 123, "target", identity),
            _derive_method_rng_metadata(7, 124, "target", identity),
            _derive_method_rng_metadata(7, 123, "other", identity),
            _derive_method_rng_metadata(
                7,
                123,
                "target",
                {"protocol": "target-v1", "config": {"n": 8}},
            ),
        ]

        self.assertTrue(
            all(
                variant["key_sha256"] != reference["key_sha256"]
                for variant in variants
            )
        )

    def test_reactivation_repeats_python_numpy_and_torch_draws(self):
        row = {}
        identity = {"protocol": "target-v1", "config": {"n": 4}}
        with mock.patch(
            "asmc_full_comparison.torch.cuda.is_available", return_value=False
        ):
            first_metadata = _activate_method_rng(
                row, 11, 9, "target", identity
            )
            first_draws = _draw_from_all_cpu_rngs()

            _draw_from_all_cpu_rngs()
            second_metadata = _activate_method_rng(
                row, 11, 9, "target", identity
            )
            second_draws = _draw_from_all_cpu_rngs()

        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_draws, second_draws)
        self.assertEqual(row["target_rng_protocol"], RNG_PROTOCOL)
        self.assertEqual(row["target_rng_seed"], first_metadata["seed"])
        self.assertEqual(
            row["target_rng_key_sha256"], first_metadata["key_sha256"]
        )
        self.assertEqual(
            row["target_rng_key_payload"], first_metadata["key_payload"]
        )

    def test_target_stream_ignores_earlier_method_and_its_config(self):
        target_identity = {
            "protocol": "target-v1",
            "config": {"temperature": 0.25},
        }

        def target_after(earlier_method, earlier_identity):
            with mock.patch(
                "asmc_full_comparison.torch.cuda.is_available",
                return_value=False,
            ):
                if earlier_method is not None:
                    _activate_method_rng(
                        {}, 19, 41, earlier_method, earlier_identity
                    )
                    _draw_from_all_cpu_rngs()
                row = {}
                metadata = _activate_method_rng(
                    row, 19, 41, "target", target_identity
                )
                return metadata, _draw_from_all_cpu_rngs(), row

        with_no_earlier_method = target_after(None, None)
        after_small = target_after(
            "earlier", {"protocol": "earlier-v1", "config": {"n": 2}}
        )
        after_large = target_after(
            "different-earlier",
            {"protocol": "earlier-v1", "config": {"n": 128}},
        )

        self.assertEqual(with_no_earlier_method, after_small)
        self.assertEqual(with_no_earlier_method, after_large)

    def test_seed_all_rngs_calls_every_backend_with_the_same_seed(self):
        with mock.patch(
            "asmc_full_comparison.random.seed"
        ) as python_seed, mock.patch(
            "asmc_full_comparison.np.random.seed"
        ) as numpy_seed, mock.patch(
            "asmc_full_comparison.torch.manual_seed"
        ) as torch_seed, mock.patch(
            "asmc_full_comparison.torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "asmc_full_comparison.torch.cuda.manual_seed_all"
        ) as cuda_seed:
            _seed_all_rngs(123456789)

        python_seed.assert_called_once_with(123456789)
        numpy_seed.assert_called_once_with(123456789)
        torch_seed.assert_called_once_with(123456789)
        cuda_seed.assert_called_once_with(123456789)


if __name__ == "__main__":
    unittest.main()
