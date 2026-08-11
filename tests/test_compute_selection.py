import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.select_compute_matched import (  # noqa: E402
    COMPARABILITY_FIELDS,
    SelectionError,
    load_summary_jsons,
    main,
    select_compute_matched,
    write_csv_selection,
    write_json_selection,
)
from analysis.result_audit import PUBLICATION_SAMPLING_PROTOCOL_METADATA  # noqa: E402


FULL_SUPPORT_SAMPLING_METADATA = {
    field: str(value)
    for field, value in PUBLICATION_SAMPLING_PROTOCOL_METADATA.items()
}


def candidate(
    method,
    config,
    accuracy,
    mean_c_int,
    p95,
    *,
    p50=1.0,
    n_problems=500,
    run_mode=None,
):
    if run_mode is None:
        run_mode = "fixed" if method == "asmc" else "single"
    return {
        "schema_version": 1,
        "method": method,
        "config": config,
        "run_mode": run_mode,
        "accuracy": accuracy,
        "mean_c_int": mean_c_int,
        "time_p50_s": p50,
        "time_p95_s": p95,
        "n_problems": n_problems,
    }


def with_provenance(summary, **overrides):
    original_config = summary["config"]
    asmc_protocol_payload = None
    asmc_protocol_sha = None
    if summary["method"] == "asmc":
        asmc_protocol_payload = json.dumps(
            {
                "backend": "batched",
                "config": {"test_label": original_config},
                "cot": True,
                "vote_mode": "weighted_no_source",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        asmc_protocol_sha = hashlib.sha256(
            asmc_protocol_payload.encode("utf-8")
        ).hexdigest()
        summary["config"] = (
            f"asmc-{summary['run_mode']}-n4-weighted_no_source-"
            f"{asmc_protocol_sha[:16]}"
        )
    elif summary["method"] == "bestofn":
        n = 2
        summary["config"] = f"n{n}_temp0.25_chunk8_lengthnorm"
    elif summary["method"] == "mcmc":
        summary["config"] = "steps2_blocks8_temp0.25"
    metadata = {
        field: {
            "code_git_dirty": "False",
            "temperature": "0.25",
            "seed": "0",
            "max_tokens": "3072",
            "trust_remote_code": "False",
            "cot": "True",
            "model_id": "Qwen/Qwen2.5-Math-7B",
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            "dataset_name": "MATH500",
            "dataset_sha256": (
                "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
            ),
            "compute_schema": "asmc-compute-v2",
            "timing_schema": "synchronized-end-to-end-wall-clock-v1",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "cuda_runtime": "12.4",
        }.get(field, f"same-{field}")
        for field in COMPARABILITY_FIELDS
    }
    metadata.update({key: str(value) for key, value in overrides.items()})
    if summary["method"] == "asmc":
        metadata.update(
            {
                "asmc_vote_mode": "weighted_no_source",
                "asmc_protocol": "cache-coherent-asmc-corrected-v1",
                "asmc_protocol_payload": asmc_protocol_payload,
                "asmc_protocol_sha256": asmc_protocol_sha,
                "asmc_use_source_weight": "False",
                "asmc_legacy_stop_constraints": "False",
                "asmc_backend": "batched",
                "asmc_use_batched": "True",
                "asmc_anneal_schedule": "cosine",
                "asmc_c_int_cap": (
                    "10" if summary["run_mode"] == "adaptive" else "none"
                ),
                "asmc_n_particles": "4",
                "asmc_fast_n_particles": "2",
                "asmc_hard_n_particles": "4",
                "asmc_block_size": "32",
                "asmc_ess_threshold": "0.5",
                "asmc_epsilon": "0.05",
                "asmc_alpha_start": "1.5",
                "asmc_alpha_star": "4.0",
                "asmc_anneal_tokens": "512",
                "asmc_early_stop_mass_threshold": "0.8",
                "asmc_early_stop_min_tokens": "64",
                "asmc_early_stop_ess_frac": "0.25",
                "asmc_early_stop_min_parsed_frac": "0.3",
                "asmc_early_stop_stable_checks": "2",
                "asmc_fast_mass_threshold": "0.65",
                "asmc_hard_anneal_tokens": "768",
                "asmc_hard_alpha_start": "1.3",
                "asmc_hard_ess_threshold": "0.6",
                "asmc_hard_epsilon": "0.08",
                "asmc_hard_early_stop_mass_threshold": "0.9",
                "asmc_hard_early_stop_min_tokens": "128",
                "asmc_hard_early_stop_ess_frac": "0.3",
                "asmc_hard_early_stop_min_parsed_frac": "0.4",
            }
        )
    elif summary["method"] == "naive":
        metadata["naive_protocol"] = "single-temperature-sample-v2"
    elif summary["method"] == "greedy":
        metadata["greedy_protocol"] = "deterministic-greedy-decoding-v2"
    elif summary["method"] == "std":
        metadata["std_protocol"] = "single-temperature-one-sample-v2"
    elif summary["method"] == "majority":
        metadata.update(
            {
                "majority_protocol": (
                    "independent-sampling-unweighted-answer-majority-v2"
                ),
                "majority_n": "4",
                "majority_temperature": "0.25",
            }
        )
    elif summary["method"] == "bestofn":
        metadata.update(
            {
                "bestofn_protocol": (
                    "independent-generation-unconditional-length-normalized-"
                    "logprob-argmax-v3"
                ),
                "bestofn_n": "2",
                "bestofn_temperature": "0.25",
                "bestofn_chunk_size": "8",
            }
        )
    elif summary["method"] == "mcmc":
        metadata.update(
            {
                "mcmc_protocol": "completion-only-eos-mcmc-power-sampling-v4",
                "mcmc_steps": "2",
                "mcmc_blocks": "8",
                "mcmc_temperature": "0.25",
            }
        )
    if summary["method"] in {
        "greedy",
        "naive",
        "std",
        "majority",
        "mcmc",
        "bestofn",
    }:
        metadata.update(FULL_SUPPORT_SAMPLING_METADATA)
    metadata.update({key: str(value) for key, value in overrides.items()})
    summary.update(
        {
            "provenance_complete": True,
            "publication_provenance_required": True,
            "invariant_metadata": metadata,
            "legacy_aliases_allowed": False,
            "_audit_summary_artifact": {
                "path": f"{summary['method']}-{summary['config']}.summary.json",
                "sha256": "a" * 64,
                "audit_schema_version": 1,
            },
        }
    )
    return summary


class ComputeMatchedSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_summary(self, name, summary):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        return path

    def test_selects_best_configuration_inside_each_tolerated_cap(self):
        summaries = [
            candidate("asmc", "small", 0.75, 101.0, 10.0),
            candidate("asmc", "large", 0.82, 103.0, 12.0),
            candidate("bestofn", "n2", 0.70, 99.0, 11.0),
            candidate("bestofn", "n4", 0.80, 109.0, 20.0),
        ]

        report = select_compute_matched(summaries, [110.0, 100.0])

        self.assertEqual(
            [entry["budget_c_int"] for entry in report["budgets"]],
            [100.0, 110.0],
        )
        first = {row["method"]: row for row in report["budgets"][0]["selections"]}
        second = {row["method"]: row for row in report["budgets"][1]["selections"]}
        self.assertEqual(first["asmc"]["config"], "small")
        self.assertEqual(first["bestofn"]["config"], "n2")
        self.assertEqual(second["asmc"]["config"], "large")
        self.assertEqual(second["bestofn"]["config"], "n4")
        self.assertEqual(first["asmc"]["eligible_config_count"], 1)
        self.assertAlmostEqual(report["budgets"][0]["cap_c_int"], 102.0)

    def test_tie_break_order_is_accuracy_compute_p95_then_config(self):
        summaries = [
            candidate("accuracy", "lower", 0.79, 10, 3),
            candidate("accuracy", "higher", 0.80, 20, 4),
            candidate("compute", "expensive", 0.80, 20, 2),
            candidate("compute", "cheap", 0.80, 10, 8),
            candidate("p95", "slow", 0.80, 10, 8),
            candidate("p95", "fast", 0.80, 10, 7),
            candidate("config", "zeta", 0.80, 10, 7),
            candidate("config", "alpha", 0.80, 10, 7),
        ]

        report = select_compute_matched(summaries, [100])
        selected = {
            row["method"]: row["config"]
            for row in report["budgets"][0]["selections"]
        }

        self.assertEqual(
            selected,
            {
                "accuracy": "higher",
                "compute": "cheap",
                "config": "alpha",
                "p95": "fast",
            },
        )

    def test_reports_a_method_with_no_eligible_configuration(self):
        summaries = [
            candidate("asmc", "h16", 0.8, 200, 10),
            candidate("bestofn", "n2", 0.7, 100, 10),
        ]

        report = select_compute_matched(summaries, [100], tolerance=1.0)
        rows = {row["method"]: row for row in report["budgets"][0]["selections"]}

        self.assertEqual(rows["asmc"]["status"], "no_eligible_config")
        self.assertIsNone(rows["asmc"]["config"])
        self.assertEqual(rows["bestofn"]["status"], "selected")

    def test_rejects_inconsistent_problem_counts_and_duplicate_pairs(self):
        with self.assertRaisesRegex(SelectionError, "expected 500"):
            select_compute_matched(
                [
                    candidate("asmc", "h4", 0.7, 10, 3, n_problems=500),
                    candidate("bestofn", "n2", 0.7, 10, 3, n_problems=499),
                ],
                [10],
            )

        partial = [
            candidate("asmc", "smoke", 0.7, 10, 3, n_problems=10),
            candidate("bestofn", "smoke", 0.7, 10, 3, n_problems=10),
        ]
        with self.assertRaisesRegex(SelectionError, "expected 500"):
            select_compute_matched(partial, [10])
        diagnostic = select_compute_matched(
            partial, [10], expected_problem_count=10
        )
        self.assertEqual(diagnostic["expected_problem_count"], 10)

    def test_publication_selection_requires_complete_comparable_metadata(self):
        with self.assertRaisesRegex(SelectionError, "provenance is not complete"):
            select_compute_matched(
                [candidate("asmc", "h4", 0.7, 10, 3)],
                [10],
                require_provenance=True,
            )

        summaries = [
            with_provenance(candidate("asmc", "h4", 0.7, 10, 3)),
            with_provenance(
                candidate(
                    "asmc", "adaptive-h4", 0.72, 9, 3, run_mode="adaptive"
                )
            ),
            with_provenance(candidate("bestofn", "n2", 0.8, 10, 4)),
            with_provenance(candidate("greedy", "greedy", 0.70, 6, 3)),
            with_provenance(candidate("mcmc", "m2b8", 0.74, 9, 4)),
            with_provenance(
                candidate("naive", "temp0.25", 0.7, 5, 3)
            ),
        ]
        with patch(
            "analysis.select_compute_matched._validated_source_artifacts",
            return_value=[{"path": "raw.csv", "sha256": "b" * 64}],
        ), patch("analysis.select_compute_matched._verify_audit_derivation"):
            report = select_compute_matched(
                summaries, [10], require_provenance=True
            )
        self.assertTrue(report["comparability_metadata_complete"])
        self.assertEqual(report["incomplete_comparability_fields"], [])
        self.assertEqual(report["budget_multipliers"], [2])
        self.assertEqual(report["budget_baseline"]["mean_c_int"], 5)
        self.assertEqual(
            report["required_publication_series"],
            ["asmc", "asmc-adaptive", "bestofn", "greedy", "mcmc", "naive"],
        )

        summaries[1]["invariant_metadata"]["asmc_c_int_cap"] = "20"
        with patch(
            "analysis.select_compute_matched._validated_source_artifacts",
            return_value=[{"path": "raw.csv", "sha256": "b" * 64}],
        ), patch("analysis.select_compute_matched._verify_audit_derivation"):
            wrong_instance_cap = select_compute_matched(
                summaries, [10], require_provenance=True
            )
        wrong_cap_rows = {
            row["method"]: row
            for row in wrong_instance_cap["budgets"][0]["selections"]
        }
        self.assertEqual(
            wrong_cap_rows["asmc-adaptive"]["status"],
            "no_eligible_config",
        )
        self.assertIsNone(
            wrong_cap_rows["asmc-adaptive"]["per_instance_c_int_cap"]
        )

        not_strictly_audited = with_provenance(
            candidate("asmc", "h8", 0.75, 10, 3)
        )
        not_strictly_audited["publication_provenance_required"] = False
        with self.assertRaisesRegex(SelectionError, "--require-provenance"):
            select_compute_matched(
                [not_strictly_audited], [10], require_provenance=True
            )

        incompatible = [
            with_provenance(candidate("asmc", "h4", 0.7, 10, 3)),
            with_provenance(
                candidate("bestofn", "n2", 0.8, 10, 4),
                model_revision="different-model-commit",
            ),
        ]
        with self.assertRaisesRegex(SelectionError, "incompatible run metadata"):
            select_compute_matched(incompatible, [10])

        with self.assertRaisesRegex(SelectionError, "duplicate method/config pair"):
            select_compute_matched(
                [
                    candidate("asmc", "h4", 0.7, 10, 3),
                    candidate("asmc", "h4", 0.8, 10, 3),
                ],
                [10],
            )

    def test_fixed_and_adaptive_asmc_are_distinct_series(self):
        summaries = [
            candidate("asmc", "fixed-h16", 0.78, 100, 3, run_mode="fixed"),
            candidate(
                "asmc", "adaptive-h16", 0.80, 95, 4, run_mode="adaptive"
            ),
        ]
        report = select_compute_matched(summaries, [100], tolerance=1.0)
        rows = {
            row["method"]: row for row in report["budgets"][0]["selections"]
        }
        self.assertEqual(report["methods"], ["asmc", "asmc-adaptive"])
        self.assertEqual(rows["asmc"]["run_mode"], "fixed")
        self.assertEqual(rows["asmc-adaptive"]["run_mode"], "adaptive")
        self.assertEqual(rows["asmc-adaptive"]["metric_method"], "asmc")

    def test_publication_profile_rejects_legacy_asmc_protocols(self):
        wrong_dataset = with_provenance(
            candidate("asmc", "h16", 0.8, 10, 3),
            dataset_sha256="e" * 64,
        )
        with self.assertRaisesRegex(SelectionError, "dataset_sha256"):
            select_compute_matched(
                [wrong_dataset], [10], require_provenance=True
            )

        invalid_cases = (
            ("asmc_vote_mode", "weighted"),
            ("asmc_use_source_weight", "True"),
            ("asmc_legacy_stop_constraints", "True"),
            ("asmc_backend", "sequential"),
        )
        for field, value in invalid_cases:
            with self.subTest(field=field):
                summary = with_provenance(
                    candidate("asmc", "h16", 0.8, 10, 3), **{field: value}
                )
                with self.assertRaisesRegex(SelectionError, "corrected-paper-v1"):
                    select_compute_matched(
                        [summary], [10], require_provenance=True
                    )

        uncapped_adaptive = with_provenance(
            candidate(
                "asmc", "adaptive", 0.8, 10, 3, run_mode="adaptive"
            ),
            asmc_c_int_cap="none",
        )
        with self.assertRaisesRegex(SelectionError, "positive finite"):
            select_compute_matched(
                [uncapped_adaptive], [10], require_provenance=True
            )

    def test_publication_profile_rejects_invalid_budget_or_mcmc_partition(self):
        roster = [
            with_provenance(candidate("asmc", "h4", 0.7, 10, 3)),
            with_provenance(
                candidate("asmc", "adaptive", 0.7, 10, 3, run_mode="adaptive")
            ),
            with_provenance(candidate("bestofn", "n2", 0.7, 10, 3)),
            with_provenance(candidate("greedy", "greedy", 0.7, 10, 3)),
            with_provenance(candidate("mcmc", "m2b8", 0.7, 10, 3)),
            with_provenance(candidate("naive", "temp0.25", 0.7, 5, 3)),
        ]
        with self.assertRaisesRegex(SelectionError, "budget_tolerance=1.02"):
            select_compute_matched(
                roster, [10], tolerance=1.5, require_provenance=True
            )
        with self.assertRaisesRegex(SelectionError, "exactly 500"):
            select_compute_matched(
                roster,
                [10],
                require_provenance=True,
                expected_problem_count=10,
            )
        with self.assertRaisesRegex(SelectionError, "not one of"):
            with patch(
                "analysis.select_compute_matched._validated_source_artifacts",
                return_value=[{"path": "raw.csv", "sha256": "b" * 64}],
            ), patch("analysis.select_compute_matched._verify_audit_derivation"):
                select_compute_matched(roster, [11], require_provenance=True)

        invalid_mcmc = with_provenance(
            candidate("mcmc", "m2b5", 0.7, 10, 3), mcmc_blocks=5
        )
        with self.assertRaisesRegex(SelectionError, "divide max_tokens"):
            select_compute_matched(
                [invalid_mcmc], [10], require_provenance=True
            )

    def test_publication_profile_rejects_sampling_warper_bypasses(self):
        corruptions = (
            ("sampling_policy", "legacy-policy"),
            ("sampling_use_cache", "False"),
            ("sampling_top_k", "40"),
            ("sampling_top_p", "0.95"),
            ("sampling_typical_p", "0.9"),
            ("sampling_min_p", "0.05"),
            ("sampling_epsilon_cutoff", "0.01"),
            ("sampling_eta_cutoff", "0.01"),
            ("sampling_repetition_penalty", "1.1"),
            ("sampling_no_repeat_ngram_size", "3"),
            ("sampling_bad_words_ids", "[1]"),
            ("sampling_forced_eos_token_id", "151643"),
            ("sampling_eos_token_ids", "[1]"),
            ("sampling_pad_token_id", "1"),
            ("sampling_policy_payload", "{}"),
            ("sampling_policy_sha256", "0" * 64),
        )
        for field, value in corruptions:
            with self.subTest(field=field):
                summary = with_provenance(
                    candidate("naive", "temp0.25", 0.7, 5, 3)
                )
                summary["invariant_metadata"][field] = value
                with self.assertRaisesRegex(SelectionError, field):
                    select_compute_matched(
                        [summary], [10], require_provenance=True
                    )

        missing = with_provenance(
            candidate("bestofn", "n2", 0.7, 10, 3)
        )
        del missing["invariant_metadata"]["sampling_top_p"]
        with self.assertRaisesRegex(SelectionError, "sampling_top_p"):
            select_compute_matched([missing], [10], require_provenance=True)

        old_protocols = (
            (
                "greedy",
                "greedy_protocol",
                "deterministic-greedy-decoding-v1",
            ),
            ("naive", "naive_protocol", "single-temperature-sample-v1"),
            (
                "mcmc",
                "mcmc_protocol",
                "completion-only-eos-mcmc-power-sampling-v3",
            ),
            (
                "bestofn",
                "bestofn_protocol",
                "independent-generation-unconditional-length-normalized-"
                "logprob-argmax-v2",
            ),
        )
        for method, field, old_protocol in old_protocols:
            with self.subTest(method=method):
                config = {
                    "greedy": "greedy",
                    "naive": "temp0.25",
                    "mcmc": "m2b8",
                    "bestofn": "n2",
                }[method]
                summary = with_provenance(
                    candidate(method, config, 0.7, 10, 3)
                )
                summary["invariant_metadata"][field] = old_protocol
                with self.assertRaisesRegex(SelectionError, field):
                    select_compute_matched(
                        [summary], [10], require_provenance=True
                    )

    def test_publication_roster_rejects_extra_appendix_series(self):
        roster = [
            with_provenance(candidate("asmc", "h4", 0.7, 10, 3)),
            with_provenance(
                candidate("asmc", "adaptive", 0.7, 10, 3, run_mode="adaptive")
            ),
            with_provenance(candidate("bestofn", "n2", 0.7, 10, 3)),
            with_provenance(candidate("greedy", "greedy", 0.7, 10, 3)),
            with_provenance(candidate("mcmc", "m2b8", 0.7, 10, 3)),
            with_provenance(candidate("naive", "temp0.25", 0.7, 5, 3)),
            with_provenance(candidate("std", "temp1", 0.7, 10, 3)),
        ]
        with patch(
            "analysis.select_compute_matched._validated_source_artifacts",
            return_value=[{"path": "raw.csv", "sha256": "b" * 64}],
        ), patch("analysis.select_compute_matched._verify_audit_derivation"):
            with self.assertRaisesRegex(SelectionError, "unexpected: std"):
                select_compute_matched(
                    roster, [10], require_provenance=True
                )

    def test_rejects_missing_nonfinite_and_invalid_measurements(self):
        missing = candidate("asmc", "h4", 0.7, 10, 3)
        del missing["time_p95_s"]
        with self.assertRaisesRegex(SelectionError, "missing required fields: time_p95_s"):
            select_compute_matched([missing], [10])

        wrong_schema = candidate("asmc", "h4", 0.7, 10, 3)
        wrong_schema["schema_version"] = 2
        with self.assertRaisesRegex(SelectionError, "schema_version"):
            select_compute_matched([wrong_schema], [10])

        for field, value in (
            ("accuracy", float("nan")),
            ("mean_c_int", float("inf")),
            ("time_p50_s", float("-inf")),
            ("time_p95_s", "not-a-number"),
        ):
            with self.subTest(field=field):
                broken = candidate("asmc", "h4", 0.7, 10, 3)
                broken[field] = value
                with self.assertRaises(SelectionError):
                    select_compute_matched([broken], [10])

        with self.assertRaisesRegex(SelectionError, "accuracy must be between"):
            select_compute_matched([candidate("asmc", "h4", 1.1, 10, 3)], [10])
        with self.assertRaisesRegex(SelectionError, "mean_c_int must be positive"):
            select_compute_matched([candidate("asmc", "h4", 0.7, 0, 3)], [10])
        with self.assertRaisesRegex(SelectionError, "time_p95_s must be at least"):
            select_compute_matched(
                [candidate("asmc", "h4", 0.7, 10, 2, p50=3)], [10]
            )

    def test_rejects_invalid_budgets_and_tolerance(self):
        summaries = [candidate("asmc", "h4", 0.7, 10, 3)]
        for budgets in ([], [0], [float("inf")], [10, 10]):
            with self.subTest(budgets=budgets):
                with self.assertRaises(SelectionError):
                    select_compute_matched(summaries, budgets)
        for tolerance in (0.99, float("nan"), True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(SelectionError):
                    select_compute_matched(summaries, [10], tolerance=tolerance)

    def test_json_loader_writers_and_cli(self):
        asmc_path = self.write_summary(
            "asmc.json", candidate("asmc", "h4", 0.7, 10, 3)
        )
        bon_path = self.write_summary(
            "bon.json", candidate("bestofn", "n2", 0.6, 9, 2)
        )
        summaries = load_summary_jsons([asmc_path, bon_path])
        report = select_compute_matched(summaries, [10])
        json_path = self.root / "selected.json"
        csv_path = self.root / "selected.csv"
        write_json_selection(report, json_path)
        write_csv_selection(report, csv_path)

        with json_path.open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["n_candidates"], 2)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["method"] for row in rows], ["asmc", "bestofn"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    str(asmc_path),
                    str(bon_path),
                    "--budget",
                    "10",
                    "20",
                    "--tolerance",
                    "1.0",
                ]
            )
        self.assertEqual(exit_code, 0)
        cli_report = json.loads(stdout.getvalue())
        self.assertEqual(len(cli_report["budgets"]), 2)
        self.assertEqual(cli_report["budget_tolerance"], 1.0)

    def test_loader_rejects_non_object_and_malformed_json(self):
        list_path = self.write_summary("list.json", [])
        with self.assertRaisesRegex(SelectionError, "root must be an object"):
            load_summary_jsons([list_path])

        malformed = self.root / "malformed.json"
        malformed.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "invalid summary JSON"):
            load_summary_jsons([malformed])

    def test_cli_output_paths_cannot_overwrite_inputs_or_each_other(self):
        summary_path = self.write_summary(
            "asmc.json", candidate("asmc", "h4", 0.7, 10, 3)
        )
        original = summary_path.read_bytes()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(
                [
                    str(summary_path),
                    "--budget",
                    "10",
                    "--json-out",
                    str(summary_path),
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(summary_path.read_bytes(), original)

        output = self.root / "same.out"
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(
                [
                    str(summary_path),
                    "--budget",
                    "10",
                    "--json-out",
                    str(output),
                    "--csv-out",
                    str(output),
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(output.exists())

        raw_path = self.root / "raw.csv"
        raw_path.write_text("do-not-overwrite\n", encoding="utf-8")
        raw_summary = candidate("asmc", "h4", 0.7, 10, 3)
        raw_summary["source_artifacts"] = [
            {"path": str(raw_path), "sha256": "0" * 64}
        ]
        raw_summary_path = self.write_summary("with-raw.json", raw_summary)
        original_raw = raw_path.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(
                [
                    str(raw_summary_path),
                    "--budget",
                    "10",
                    "--json-out",
                    str(raw_path),
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(raw_path.read_bytes(), original_raw)

        alias_base = self.root / "selection"
        alias_tmp = self.root / "selection.tmp"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(
                [
                    str(summary_path),
                    "--budget",
                    "10",
                    "--json-out",
                    str(alias_tmp),
                    "--csv-out",
                    str(alias_base),
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(alias_base.exists())
        self.assertFalse(alias_tmp.exists())


if __name__ == "__main__":
    unittest.main()
