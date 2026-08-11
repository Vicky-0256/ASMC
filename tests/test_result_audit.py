import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.result_audit import (
    AuditError,
    PUBLICATION_METADATA_COLUMNS,
    audit_csvs,
    main as audit_main,
    write_csv_summary,
    write_json_summary,
)
from asmc_sampler import ASMCConfig
from asmc_full_comparison import RNG_PROTOCOL, _derive_method_rng_metadata


FIELDNAMES = [
    "problem_idx",
    "batch_idx",
    "question",
    "correct_answer",
    "rng_protocol",
    "method",
    "config",
    "asmc_correct",
    "asmc_answer",
    "asmc_completion",
    "asmc_completion_token_ids",
    "asmc_completion_has_eos",
    "asmc_time_s",
    "asmc_c_int",
    "asmc_c_tok",
    "asmc_c_step",
    "asmc_n_forward",
    "asmc_prefill_flops",
    "asmc_decode_flops",
    "asmc_total_flops",
    "asmc_pass_type",
    "asmc_mode",
    "asmc_config",
    "asmc_protocol",
    "asmc_protocol_sha256",
    "asmc_protocol_payload",
    "asmc_rng_protocol",
    "asmc_rng_seed",
    "asmc_rng_key_sha256",
    "asmc_rng_key_payload",
    "asmc_backend",
    "asmc_use_batched",
    "asmc_vote_mode",
    "asmc_use_source_weight",
    "asmc_c_int_cap",
    "asmc_budget_exhausted",
    "asmc_budget_exhausted_at_token",
    "asmc_stop_reason",
    "asmc_legacy_stop_constraints",
    "asmc_n_particles",
    "asmc_fast_n_particles",
    "asmc_hard_n_particles",
    "asmc_block_size",
    "asmc_ess_threshold",
    "asmc_epsilon",
    "asmc_alpha_start",
    "asmc_alpha_star",
    "asmc_anneal_tokens",
    "asmc_anneal_schedule",
    "asmc_early_stop_mass_threshold",
    "asmc_early_stop_min_tokens",
    "asmc_early_stop_ess_frac",
    "asmc_early_stop_min_parsed_frac",
    "asmc_early_stop_stable_checks",
    "asmc_fast_mass_threshold",
    "asmc_hard_anneal_tokens",
    "asmc_hard_alpha_start",
    "asmc_hard_ess_threshold",
    "asmc_hard_epsilon",
    "asmc_hard_early_stop_mass_threshold",
    "asmc_hard_early_stop_min_tokens",
    "asmc_hard_early_stop_ess_frac",
    "asmc_hard_early_stop_min_parsed_frac",
    *PUBLICATION_METADATA_COLUMNS,
]

MATH500_ROWS = json.loads((REPOSITORY_ROOT / "data" / "MATH500.json").read_text())


def attach_canonical_asmc_protocol(
    rows, *, c_int_cap=None, adaptive=False, backend="batched"
):
    """Attach the exact content-addressed ASMCConfig used by strict audit."""

    config = ASMCConfig(
        # The public runner receives this through an argparse ``float`` flag;
        # mirror that resolved runtime type even when a test passes ``5``.
        c_int_cap=None if c_int_cap is None else float(c_int_cap),
        n_particles=4,
        max_new_tokens=3072,
        early_stop_min_tokens=64,
        enable_adaptive=adaptive,
        stop_token_ids=[151643],
    )
    payload = json.dumps(
        {
            "backend": backend,
            "config": vars(config),
            "cot": True,
            "vote_mode": "weighted_no_source",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    run_mode = "adaptive" if adaptive else "fixed"
    config_id = (
        f"asmc-{run_mode}-n{config.n_particles}-weighted_no_source-"
        f"{payload_sha[:16]}"
    )
    for row in rows:
        row["config"] = config_id
        row["asmc_config"] = config_id
        row["asmc_backend"] = backend
        row["asmc_use_batched"] = backend == "batched"
        row["asmc_protocol_payload"] = payload
        row["asmc_protocol_sha256"] = payload_sha
    attach_asmc_rng(rows, config_id=config_id)
    return config_id


def attach_asmc_rng(rows, *, config_id=None):
    for row in rows:
        resolved_config_id = config_id or row["asmc_config"]
        common = {
            "attn_implementation": row["attn_implementation"],
            "cot": bool(row["cot"]),
            "dataset_name": row["dataset_name"],
            "dataset_sha256": row["dataset_sha256"],
            "dtype": row["dtype"],
            "max_new_tokens": int(row["max_tokens"]),
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
        }
        method_identity = {
            "common": common,
            "config_id": resolved_config_id,
            "protocol": row["asmc_protocol"],
            "protocol_payload": json.loads(row["asmc_protocol_payload"]),
        }
        metadata = _derive_method_rng_metadata(
            int(row["seed"]), int(row["problem_idx"]), "asmc", method_identity
        )
        row["rng_protocol"] = RNG_PROTOCOL
        row["asmc_rng_protocol"] = metadata["protocol"]
        row["asmc_rng_seed"] = metadata["seed"]
        row["asmc_rng_key_sha256"] = metadata["key_sha256"]
        row["asmc_rng_key_payload"] = metadata["key_payload"]


class ResultAuditTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_rows(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_cli_output_paths_cannot_overwrite_inputs_or_each_other(self):
        source = self.write_rows("input.csv", [])
        original = source.read_bytes()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = audit_main(
                [
                    str(source),
                    "--config",
                    "fixed",
                    "--mode",
                    "fixed",
                    "--json-out",
                    str(source),
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(source.read_bytes(), original)
        self.assertIn("must not overwrite", stderr.getvalue())

        output = self.root / "same.out"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = audit_main(
                [
                    str(source),
                    "--config",
                    "fixed",
                    "--mode",
                    "fixed",
                    "--json-out",
                    str(output),
                    "--csv-out",
                    str(output),
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertFalse(output.exists())

        alias_base = self.root / "summary"
        alias_tmp = self.root / "summary.tmp"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = audit_main(
                [
                    str(source),
                    "--config",
                    "fixed",
                    "--mode",
                    "fixed",
                    "--json-out",
                    str(alias_tmp),
                    "--csv-out",
                    str(alias_base),
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertFalse(alias_base.exists())
        self.assertFalse(alias_tmp.exists())

    @staticmethod
    def row(problem_idx, correct, time_s, c_int, pass_type="single"):
        return {
            "problem_idx": problem_idx,
            "method": "asmc",
            "config": "h4_ann512",
            "asmc_correct": correct,
            "asmc_time_s": time_s,
            "asmc_c_int": c_int,
            "asmc_pass_type": pass_type,
        }

    def test_fixed_multi_file_summary_and_writers(self):
        first = self.write_rows(
            "batch0.csv",
            [self.row(0, True, 1, 10), self.row(1, False, 2, 20)],
        )
        second = self.write_rows(
            "batch1.csv",
            [
                self.row(2, True, 3, 30),
                self.row(3, False, 4, 40),
                self.row(4, True, 5, 50),
            ],
        )

        summary = audit_csvs(
            [first, second],
            method="asmc",
            config="h4_ann512",
            run_mode="fixed",
            expected_problem_count=5,
        )

        self.assertEqual(summary["n_files"], 2)
        self.assertEqual(summary["n_problems"], 5)
        self.assertAlmostEqual(summary["accuracy"], 0.6)
        self.assertAlmostEqual(summary["time_p50_s"], 3.0)
        self.assertAlmostEqual(summary["time_p95_s"], 4.8)
        self.assertAlmostEqual(summary["mean_c_int"], 30.0)
        self.assertEqual(summary["pass_type_counts"], {"single": 5})
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(len(summary["source_artifacts"]), 2)
        self.assertEqual(len(summary["source_artifacts"][0]["sha256"]), 64)
        self.assertNotIn("git_commit", summary)
        self.assertNotIn("code_sha", summary)

        json_path = self.root / "summary.json"
        csv_path = self.root / "summary.csv"
        write_json_summary(summary, json_path)
        write_csv_summary(summary, csv_path)
        with json_path.open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["config"], "h4_ann512")
        with csv_path.open(encoding="utf-8", newline="") as handle:
            csv_summary = next(csv.DictReader(handle))
        self.assertEqual(csv_summary["method"], "asmc")
        self.assertEqual(json.loads(csv_summary["pass_type_counts"]), {"single": 5})

    def test_method_and_run_mode_semantics_are_enforced(self):
        path = self.write_rows("mode.csv", [self.row(0, True, 1, 10)])
        with self.assertRaisesRegex(AuditError, "ASMC run_mode"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="single",
                expected_problem_count=1,
            )
        with self.assertRaisesRegex(AuditError, "non-ASMC run_mode"):
            audit_csvs(
                [path],
                method="bestofn",
                config="n4",
                run_mode="fixed",
                expected_problem_count=1,
            )

    def test_numeric_audit_arguments_reject_bool_nan_and_infinity(self):
        path = self.write_rows("arguments.csv", [self.row(0, True, 1, 10)])
        for invalid_count in (True, 1.5):
            with self.subTest(expected_problem_count=invalid_count):
                with self.assertRaisesRegex(AuditError, "positive integer"):
                    audit_csvs(
                        [path],
                        method="asmc",
                        config="h4_ann512",
                        run_mode="fixed",
                        expected_problem_count=invalid_count,
                    )
        for invalid_cap in (True, float("nan"), float("inf")):
            with self.subTest(budget_cap=invalid_cap):
                with self.assertRaisesRegex(AuditError, "finite and positive"):
                    audit_csvs(
                        [path],
                        method="asmc",
                        config="h4_ann512",
                        run_mode="fixed",
                        expected_problem_count=1,
                        budget_cap=invalid_cap,
                    )
        for invalid_tolerance in (True, float("nan"), float("inf")):
            with self.subTest(budget_tolerance=invalid_tolerance):
                with self.assertRaisesRegex(AuditError, "finite and at least"):
                    audit_csvs(
                        [path],
                        method="asmc",
                        config="h4_ann512",
                        run_mode="fixed",
                        expected_problem_count=1,
                        budget_tolerance=invalid_tolerance,
                    )
        with self.assertRaisesRegex(AuditError, "exactly 500"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=10,
                require_provenance=True,
            )

    def test_reports_duplicate_and_missing_problem_ids_together(self):
        path = self.write_rows(
            "broken.csv",
            [
                self.row(0, True, 1, 10),
                self.row(1, True, 1, 10),
                self.row(1, True, 1, 10),
                self.row(3, True, 1, 10),
                self.row(4, True, 1, 10),
            ],
        )
        with self.assertRaises(AuditError) as raised:
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=5,
            )
        self.assertIn("duplicate problem_idx: 1", str(raised.exception))
        self.assertIn("missing problem_idx: [2]", str(raised.exception))

    def test_fixed_rejects_adaptive_pass_type(self):
        path = self.write_rows(
            "mixed.csv",
            [self.row(i, True, 1, 10, "fast" if i == 2 else "single") for i in range(5)],
        )
        with self.assertRaisesRegex(AuditError, "fixed run requires"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=5,
            )

    def test_adaptive_accepts_fast_and_hard_but_not_single(self):
        valid = self.write_rows(
            "adaptive.csv",
            [
                self.row(i, True, i + 1, 10, "hard" if i == 4 else "fast")
                for i in range(5)
            ],
        )
        summary = audit_csvs(
            [valid],
            method="asmc",
            config="h4_ann512",
            run_mode="adaptive",
            expected_problem_count=5,
        )
        self.assertEqual(summary["pass_type_counts"], {"fast": 4, "hard": 1})

        invalid = self.write_rows(
            "not_adaptive.csv",
            [self.row(i, True, 1, 10, "single") for i in range(5)],
        )
        with self.assertRaisesRegex(AuditError, "adaptive run requires"):
            audit_csvs(
                [invalid],
                method="asmc",
                config="h4_ann512",
                run_mode="adaptive",
                expected_problem_count=5,
            )

    def test_single_mode_audits_a_baseline_prefix(self):
        fields = [
            "problem_idx",
            "bestofn_config",
            "bestofn_mode",
            "bestofn_correct",
            "bestofn_time_s",
            "bestofn_c_int",
            "bestofn_pass_type",
        ]
        path = self.root / "bestofn.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for problem_idx in range(2):
                writer.writerow(
                    {
                        "problem_idx": problem_idx,
                        "bestofn_config": "n4_temp0.25_chunk8_lengthnorm",
                        "bestofn_mode": "single",
                        "bestofn_correct": problem_idx == 0,
                        "bestofn_time_s": problem_idx + 1,
                        "bestofn_c_int": 10 * (problem_idx + 1),
                        "bestofn_pass_type": "single",
                    }
                )

        summary = audit_csvs(
            [path],
            method="bestofn",
            config="n4_temp0.25_chunk8_lengthnorm",
            run_mode="single",
            expected_problem_count=2,
        )
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["run_mode"], "single")
        self.assertEqual(summary["pass_type_counts"], {"single": 2})

    def test_checks_method_and_config_metadata_when_present(self):
        wrong_method = self.row(0, True, 1, 10)
        wrong_method["method"] = "bestofn"
        path = self.write_rows("wrong_method.csv", [wrong_method])
        with self.assertRaisesRegex(AuditError, "method metadata"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=1,
            )

        wrong_config = self.row(0, True, 1, 10)
        wrong_config["config"] = "h16_ann512"
        path = self.write_rows("wrong_config.csv", [wrong_config])
        with self.assertRaisesRegex(AuditError, "config metadata"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=1,
            )

    def test_legacy_metric_aliases_require_explicit_opt_in(self):
        legacy_fields = [
            "problem_idx",
            "method",
            "config",
            "asmc_correct",
            "asmc_time",
            "asmc_total_flops",
            "asmc_pass_type",
        ]
        path = self.root / "legacy.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=legacy_fields)
            writer.writeheader()
            for problem_idx in range(2):
                writer.writerow(
                    {
                        "problem_idx": problem_idx,
                        "method": "asmc",
                        "config": "h4_ann512",
                        "asmc_correct": problem_idx == 0,
                        "asmc_time": problem_idx + 1,
                        "asmc_total_flops": (problem_idx + 1) * 10,
                        "asmc_pass_type": "single",
                    }
                )

        with self.assertRaisesRegex(AuditError, "allow-legacy-aliases"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=2,
            )

        summary = audit_csvs(
            [path],
            method="asmc",
            config="h4_ann512",
            run_mode="fixed",
            expected_problem_count=2,
            allow_legacy_aliases=True,
        )
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["mean_c_int"], 15.0)
        self.assertEqual(
            summary["metric_column_usage"]["c_int"],
            {"asmc_total_flops": 2},
        )
        with self.assertRaisesRegex(AuditError, "forbids legacy metric aliases"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=2,
                allow_legacy_aliases=True,
                require_provenance=True,
            )

    def test_optional_mean_compute_budget_is_enforced(self):
        path = self.write_rows(
            "budget.csv",
            [self.row(0, True, 1, 100), self.row(1, True, 1, 110)],
        )
        with self.assertRaisesRegex(AuditError, "exceeds budget limit"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=2,
                budget_cap=100,
                budget_tolerance=1.02,
            )

        summary = audit_csvs(
            [path],
            method="asmc",
            config="h4_ann512",
            run_mode="fixed",
            expected_problem_count=2,
            budget_cap=105,
            budget_tolerance=1.02,
        )
        self.assertAlmostEqual(summary["budget_limit"], 107.1)

    def test_c_int_must_be_strictly_positive(self):
        path = self.write_rows("zero_compute.csv", [self.row(0, True, 0, 0)])
        with self.assertRaisesRegex(AuditError, "asmc_c_int must be positive"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=1,
            )

    def test_publication_gate_requires_complete_clean_provenance(self):
        incomplete_row = self.row(0, True, 1, 10)
        incomplete_row.update(
            {
                "asmc_budget_exhausted": False,
                "asmc_budget_exhausted_at_token": "",
                "asmc_stop_reason": "max_len",
            }
        )
        incomplete = self.write_rows(
            "incomplete.csv", [incomplete_row]
        )
        with self.assertRaises(AuditError):
            audit_csvs(
                [incomplete],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=1,
                require_provenance=True,
            )

        rows = []
        for problem_idx in range(500):
            benchmark_row = MATH500_ROWS[problem_idx]
            row = self.row(problem_idx, problem_idx == 0, 1, 10)
            row.update(
                {
                    "batch_idx": problem_idx // 100,
                    "question": benchmark_row["prompt"],
                    "correct_answer": benchmark_row["answer"],
                    "asmc_answer": (
                        ""
                        if problem_idx == 1
                        else benchmark_row["answer"]
                        if problem_idx == 0
                        else "__definitely_incorrect__"
                    ),
                    "asmc_completion": (
                        ""
                        if problem_idx == 1
                        else "The final answer is \\boxed{"
                        + (
                            benchmark_row["answer"]
                            if problem_idx == 0
                            else "__definitely_incorrect__"
                        )
                        + "}."
                    ),
                    "asmc_completion_token_ids": (
                        "[151643]" if problem_idx == 1 else "[42,151643]"
                    ),
                    "asmc_completion_has_eos": True,
                    "asmc_config": "h4_ann512",
                    "asmc_protocol": "cache-coherent-asmc-corrected-v1",
                    "asmc_mode": "fixed",
                    "asmc_backend": "batched",
                    "asmc_use_batched": True,
                    "asmc_vote_mode": "weighted_no_source",
                    "asmc_use_source_weight": False,
                    "asmc_c_int_cap": "none",
                    "asmc_c_tok": 3,
                    "asmc_c_step": 2,
                    "asmc_n_forward": 1,
                    "asmc_prefill_flops": 4,
                    "asmc_decode_flops": 6,
                    "asmc_total_flops": 10,
                    "asmc_budget_exhausted": False,
                    "asmc_budget_exhausted_at_token": "",
                    "asmc_stop_reason": "max_len",
                    "asmc_legacy_stop_constraints": False,
                    "asmc_n_particles": 4,
                    "asmc_fast_n_particles": 2,
                    "asmc_hard_n_particles": 4,
                    "asmc_block_size": 32,
                    "asmc_ess_threshold": 0.5,
                    "asmc_epsilon": 0.05,
                    "asmc_alpha_start": 1.5,
                    "asmc_alpha_star": 4.0,
                    "asmc_anneal_tokens": 512,
                    "asmc_anneal_schedule": "cosine",
                    "asmc_early_stop_mass_threshold": 0.8,
                    "asmc_early_stop_min_tokens": 64,
                    "asmc_early_stop_ess_frac": 0.25,
                    "asmc_early_stop_min_parsed_frac": 0.3,
                    "asmc_early_stop_stable_checks": 2,
                    "asmc_fast_mass_threshold": 0.65,
                    "asmc_hard_anneal_tokens": 768,
                    "asmc_hard_alpha_start": 1.3,
                    "asmc_hard_ess_threshold": 0.6,
                    "asmc_hard_epsilon": 0.08,
                    "asmc_hard_early_stop_mass_threshold": 0.9,
                    "asmc_hard_early_stop_min_tokens": 128,
                    "asmc_hard_early_stop_ess_frac": 0.3,
                    "asmc_hard_early_stop_min_parsed_frac": 0.4,
                    "model_id": "Qwen/Qwen2.5-Math-7B",
                    "model_revision": "a" * 40,
                    "dtype": "bfloat16",
                    "attn_implementation": "flash_attention_2",
                    "trust_remote_code": False,
                    "dataset_name": "MATH500",
                    "dataset_sha256": (
                        "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38"
                        "fc500c6561e1e06"
                    ),
                    "code_git_commit": "c" * 40,
                    "code_git_dirty": False,
                    "python_version": "3.11.9",
                    "pytorch_version": "2.5.1",
                    "transformers_version": "4.47.1",
                    "cuda_runtime": "12.4",
                    "gpu_name": "NVIDIA A100-SXM4-80GB",
                    "nvidia_driver_version": "550.54.15",
                    "flash_attn_version": "2.7.4",
                    "cot": True,
                    "max_tokens": 3072,
                    "temperature": 0.25,
                    "seed": 0,
                    "compute_schema": "asmc-compute-v2",
                    "timing_schema": "synchronized-end-to-end-wall-clock-v1",
                }
            )
            rows.append(row)
        attach_canonical_asmc_protocol(rows)
        complete = self.write_rows("publication.csv", rows)

        summary = audit_csvs(
            [complete],
            method="asmc",
            config=rows[0]["asmc_config"],
            run_mode="fixed",
            expected_problem_count=500,
            require_provenance=True,
        )
        self.assertTrue(summary["provenance_complete"])
        self.assertEqual(summary["incomplete_publication_columns"], [])
        self.assertEqual(
            summary["correctness_evidence_counts"],
            {"completion": 499, "completion_token_ids": 1},
        )

        for row in rows:
            row["config"] = "h4_ann512"
            row["asmc_config"] = "h4_ann512"
        attach_asmc_rng(rows, config_id="h4_ann512")
        short_alias = self.write_rows("short_alias.csv", rows)
        with self.assertRaisesRegex(AuditError, "content-addressed runner ID"):
            audit_csvs(
                [short_alias],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )
        attach_canonical_asmc_protocol(rows)

        for row in rows:
            row["asmc_mode"] = "adaptive"
            row["asmc_pass_type"] = "fast"
        attach_canonical_asmc_protocol(rows, adaptive=True)
        adaptive_without_cap = self.write_rows("adaptive_without_cap.csv", rows)
        with self.assertRaisesRegex(AuditError, "finite positive asmc_c_int_cap"):
            audit_csvs(
                [adaptive_without_cap],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="adaptive",
                expected_problem_count=500,
                require_provenance=True,
            )
        for row in rows:
            row["asmc_mode"] = "fixed"
            row["asmc_pass_type"] = "single"
        attach_canonical_asmc_protocol(rows)

        attach_canonical_asmc_protocol(rows, backend="sequential")
        sequential_publication = self.write_rows("sequential.csv", rows)
        with self.assertRaisesRegex(AuditError, "cache-coherent batched backend"):
            audit_csvs(
                [sequential_publication],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )
        attach_canonical_asmc_protocol(rows)

        for row in rows:
            row["model_id"] = "Qwen/Qwen2.5-7B"
        attach_asmc_rng(rows)
        wrong_model = self.write_rows("wrong_model.csv", rows)
        with self.assertRaisesRegex(AuditError, "model_id"):
            audit_csvs(
                [wrong_model],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )
        for row in rows:
            row["model_id"] = "Qwen/Qwen2.5-Math-7B"
        attach_asmc_rng(rows)

        canonical_protocol_payload = rows[0]["asmc_protocol_payload"]
        canonical_protocol_sha = rows[0]["asmc_protocol_sha256"]
        noncanonical_protocol_payload = json.dumps(
            json.loads(canonical_protocol_payload), indent=2
        )
        noncanonical_protocol_sha = hashlib.sha256(
            noncanonical_protocol_payload.encode("utf-8")
        ).hexdigest()
        for row in rows:
            row["asmc_protocol_payload"] = noncanonical_protocol_payload
            row["asmc_protocol_sha256"] = noncanonical_protocol_sha
        noncanonical = self.write_rows("noncanonical_protocol.csv", rows)
        with self.assertRaisesRegex(AuditError, "canonical sorted compact JSON"):
            audit_csvs(
                [noncanonical],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )
        for row in rows:
            row["asmc_protocol_payload"] = canonical_protocol_payload
            row["asmc_protocol_sha256"] = canonical_protocol_sha

        rows[0]["asmc_total_flops"] = 1_000_000
        mismatched_components = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "must equal"):
            audit_csvs(
                [mismatched_components],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )
        rows[0]["asmc_total_flops"] = 10

        # pandas promotes an optional integer column to float when some rows
        # are missing, so real shards serialize token indices as e.g. "0.0".
        rows[0]["asmc_c_int_cap"] = 5
        rows[0]["asmc_c_int"] = 10
        rows[0]["asmc_budget_exhausted"] = True
        rows[0]["asmc_budget_exhausted_at_token"] = "0.0"
        rows[0]["asmc_stop_reason"] = "budget_exhausted"
        for row in rows[1:]:
            row["asmc_c_int_cap"] = 5
            row["asmc_c_int"] = 4
            row["asmc_prefill_flops"] = 2
            row["asmc_decode_flops"] = 2
            row["asmc_total_flops"] = 4
            row["asmc_budget_exhausted"] = False
            row["asmc_budget_exhausted_at_token"] = ""
            row["asmc_stop_reason"] = "max_len"
        attach_canonical_asmc_protocol(rows, c_int_cap=5)
        pandas_optional_integer = self.write_rows("publication.csv", rows)
        pandas_summary = audit_csvs(
            [pandas_optional_integer],
            method="asmc",
            config=rows[0]["asmc_config"],
            run_mode="fixed",
            expected_problem_count=500,
            require_provenance=True,
        )
        self.assertTrue(pandas_summary["provenance_complete"])

        for row in rows:
            row["asmc_c_int"] = 10
            row["asmc_prefill_flops"] = 4
            row["asmc_decode_flops"] = 6
            row["asmc_total_flops"] = 10
            row["asmc_c_int_cap"] = "none"
            row["asmc_budget_exhausted"] = False
            row["asmc_budget_exhausted_at_token"] = ""
            row["asmc_stop_reason"] = "max_len"
        attach_canonical_asmc_protocol(rows)

        for row in rows:
            row["compute_schema"] = "wrong-compute"
        wrong_compute_schema = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "compute_schema"):
            audit_csvs(
                [wrong_compute_schema],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["compute_schema"] = "asmc-compute-v2"
            row["timing_schema"] = "wrong-timing"
        wrong_timing_schema = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "timing_schema"):
            audit_csvs(
                [wrong_timing_schema],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["timing_schema"] = "synchronized-end-to-end-wall-clock-v1"
            row["asmc_c_int_cap"] = 5
            row["asmc_budget_exhausted"] = False
        attach_canonical_asmc_protocol(rows, c_int_cap=5)
        unreported_cap = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "reached/exceeded"):
            audit_csvs(
                [unreported_cap],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["asmc_c_int_cap"] = "none"
            row["model_revision"] = "mutable-tag"
        attach_canonical_asmc_protocol(rows)
        mutable_revision = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "immutable"):
            audit_csvs(
                [mutable_revision],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["model_revision"] = "a" * 40
            row["asmc_use_source_weight"] = True
        attach_asmc_rng(rows)
        inconsistent_vote = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "use_source_weight is inconsistent"):
            audit_csvs(
                [inconsistent_vote],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["asmc_use_source_weight"] = False
            row["asmc_c_int_cap"] = 100
            row["asmc_budget_exhausted"] = True
            row["asmc_budget_exhausted_at_token"] = 0
            row["asmc_stop_reason"] = "max_len"
        attach_canonical_asmc_protocol(rows, c_int_cap=100)
        inconsistent_budget = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "exhausted budget requires"):
            audit_csvs(
                [inconsistent_budget],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["asmc_c_int_cap"] = "none"
            row["asmc_budget_exhausted"] = False
            row["asmc_budget_exhausted_at_token"] = ""
            row["asmc_stop_reason"] = "max_len"
            row["nvidia_driver_version"] = "unknown"
        attach_canonical_asmc_protocol(rows)
        unknown_driver = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "resolved nvidia_driver_version"):
            audit_csvs(
                [unknown_driver],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

        for row in rows:
            row["nvidia_driver_version"] = "550.54.15"
            row["flash_attn_version"] = "not-installed"
        missing_flash = self.write_rows("publication.csv", rows)
        with self.assertRaisesRegex(AuditError, "resolved flash_attn_version"):
            audit_csvs(
                [missing_flash],
                method="asmc",
                config=rows[0]["asmc_config"],
                run_mode="fixed",
                expected_problem_count=500,
                require_provenance=True,
            )

    def test_rejects_mixed_invariant_metadata(self):
        first = self.row(0, True, 1, 10)
        second = self.row(1, True, 1, 10)
        first.update({"seed": 0, "asmc_mode": "fixed"})
        second.update({"seed": 1, "asmc_mode": "fixed"})
        path = self.write_rows("mixed_seed.csv", [first, second])

        with self.assertRaisesRegex(AuditError, "invariant metadata seed"):
            audit_csvs(
                [path],
                method="asmc",
                config="h4_ann512",
                run_mode="fixed",
                expected_problem_count=2,
            )


if __name__ == "__main__":
    unittest.main()
