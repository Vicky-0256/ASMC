"""CPU-only tests for deterministic paper-table rendering."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.render_compute_table import (
    RenderError,
    render_outputs,
    validate_selection_report,
)
from analysis.select_compute_matched import (
    SelectionError,
    load_summary_jsons,
    select_compute_matched,
)
from analysis.result_audit import (
    PUBLICATION_SAMPLING_PROTOCOL_METADATA,
    _build_method_rng_identity,
    _grade_with_runner_fallback,
    audit_csvs,
)
from asmc_full_comparison import RNG_PROTOCOL, _derive_method_rng_metadata
from asmc_sampler import ASMCConfig
from grader_utils.math_grader import grade_answer
from grader_utils.parse_utils import parse_answer, parse_answer_robust


def _asmc_protocol_identity(run_mode):
    adaptive = run_mode == "adaptive"
    config = ASMCConfig(
        c_int_cap=1_000_000.0 if adaptive else None,
        n_particles=4,
        max_new_tokens=3072,
        early_stop_min_tokens=64,
        enable_adaptive=adaptive,
        stop_token_ids=[151643],
    )
    payload = json.dumps(
        {
            "backend": "batched",
            "config": vars(config),
            "cot": True,
            "vote_mode": "weighted_no_source",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    protocol_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    config_id = (
        f"asmc-{run_mode}-n4-weighted_no_source-{protocol_sha[:16]}"
    )
    return payload, protocol_sha, config_id


ASMC_PROTOCOL_PAYLOAD, ASMC_PROTOCOL_SHA, ASMC_TEST_CONFIG = (
    _asmc_protocol_identity("fixed")
)
(
    ASMC_ADAPTIVE_PROTOCOL_PAYLOAD,
    ASMC_ADAPTIVE_PROTOCOL_SHA,
    ASMC_ADAPTIVE_TEST_CONFIG,
) = _asmc_protocol_identity("adaptive")
MATH500_ROWS = json.loads((PROJECT_ROOT / "data" / "MATH500.json").read_text())


def _report():
    specs = [
        ("asmc", "asmc", "fixed", ASMC_TEST_CONFIG, 0.75, 900000.0, 2.0, 4.0),
        (
            "asmc-adaptive",
            "asmc",
            "adaptive",
            ASMC_ADAPTIVE_TEST_CONFIG,
            0.76,
            850000.0,
            2.0,
            4.0,
        ),
        (
            "bestofn",
            "bestofn",
            "single",
            "n2_temp0.25_chunk8_lengthnorm",
            0.73,
            800000.0,
            2.0,
            4.0,
        ),
        ("greedy", "greedy", "single", "greedy", 0.70, 1100000.0, 3.0, 3.0),
        (
            "mcmc",
            "mcmc",
            "single",
            "steps2_blocks8_temp0.25",
            0.72,
            950000.0,
            2.0,
            4.0,
        ),
        ("naive", "naive", "single", "temp0.25", 0.70, 500000.0, 3.0, 3.0),
    ]
    input_summaries = []
    selections = []
    for index, (
        series,
        metric_method,
        run_mode,
        config,
        accuracy,
        mean_c_int,
        p50,
        p95,
    ) in enumerate(specs, start=1):
        input_summaries.append(
            {
                "path": f"{series}.summary.json",
                "sha256": f"{index:064x}",
                "audit_schema_version": 1,
                "method": series,
                "metric_method": metric_method,
                "run_mode": run_mode,
                "config": config,
                "source_artifacts": [
                    {"path": f"{series}.csv", "sha256": f"{index + 16:064x}"}
                ],
            }
        )
        eligible = mean_c_int <= 1_020_000.0
        selections.append(
            {
                "method": series,
                "metric_method": metric_method,
                "run_mode": run_mode,
                "status": "selected" if eligible else "no_eligible_config",
                "eligible_config_count": 1 if eligible else 0,
                "config": config if eligible else None,
                "accuracy": accuracy if eligible else None,
                "mean_c_int": mean_c_int if eligible else None,
                "time_p50_s": p50 if eligible else None,
                "time_p95_s": p95 if eligible else None,
                "per_instance_c_int_cap": (
                    1_000_000.0
                    if series == "asmc-adaptive" and eligible
                    else None
                ),
                "n_problems": 500,
            }
        )
    return {
        "schema_version": 1,
        "budget_tolerance": 1.02,
        "n_problems": 500,
        "expected_problem_count": 500,
        "n_candidates": len(specs),
        "input_summaries": input_summaries,
        "methods": [spec[0] for spec in specs],
        "required_publication_series": [
            "asmc",
            "asmc-adaptive",
            "bestofn",
            "greedy",
            "mcmc",
            "naive",
        ],
        "budget_baseline": {
            "method": "naive",
            "config": "temp0.25",
            "mean_c_int": 500000.0,
        },
        "budget_multipliers": [2],
        "publication_provenance_required": True,
        "raw_audit_derivation_verified": True,
        "asmc_publication_profile": "corrected-paper-v1",
        "comparability_metadata_complete": True,
        "common_invariant_metadata": {"code_git_commit": "abc123"},
        "budgets": [
            {
                "budget_c_int": 1000000.0,
                "cap_c_int": 1020000.0,
                "selections": selections,
            }
        ],
    }


def _common_metadata():
    return {
        "model_id": "Qwen/Qwen2.5-Math-7B",
        "model_revision": "a" * 40,
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": False,
        "dataset_name": "MATH500",
        "dataset_sha256": (
            "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
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
        **PUBLICATION_SAMPLING_PROTOCOL_METADATA,
    }


def _asmc_metadata(run_mode="fixed"):
    return {
        "asmc_backend": "batched",
        "asmc_use_batched": True,
        "asmc_vote_mode": "weighted_no_source",
        "asmc_use_source_weight": False,
        "asmc_c_int_cap": 1_000_000 if run_mode == "adaptive" else "none",
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
    }


def _write_audited_summary(
    root, method, config, accuracy, mean_c_int, p95, *, run_mode=None
):
    if run_mode is None:
        run_mode = "fixed" if method == "asmc" else "single"
    series = "asmc-adaptive" if method == "asmc" and run_mode == "adaptive" else method
    raw_path = root / f"{series}.csv"
    rows = []
    n_correct = round(500 * accuracy)
    parser = parse_answer_robust if method == "asmc" else parse_answer
    candidate_artifacts = []
    for benchmark_row in MATH500_ROWS:
        completion = (
            f"The final answer is \\boxed{{{benchmark_row['answer']}}}."
        )
        parsed_answer = parser(completion)
        can_grade_correctly = _grade_with_runner_fallback(
            parsed_answer, benchmark_row["answer"], grade_answer=grade_answer
        )
        candidate_artifacts.append(
            (completion, parsed_answer, can_grade_correctly)
        )
    correct_indices = {
        index
        for index, (_, _, can_grade_correctly) in enumerate(candidate_artifacts)
        if can_grade_correctly
    }
    correct_indices = set(sorted(correct_indices)[:n_correct])
    if len(correct_indices) != n_correct:
        raise AssertionError("fixture cannot construct the requested accuracy")
    for problem_idx in range(500):
        benchmark_row = MATH500_ROWS[problem_idx]
        expected_correct = problem_idx in correct_indices
        if expected_correct:
            emitted_completion, parsed_answer, _ = candidate_artifacts[problem_idx]
            emitted_answer = str(parsed_answer).strip()
        else:
            emitted_answer = "__definitely_incorrect__"
            emitted_completion = (
                f"The final answer is \\boxed{{{emitted_answer}}}."
            )
        row = {
            "problem_idx": problem_idx,
            "batch_idx": problem_idx // 100,
            "question": benchmark_row["prompt"],
            "correct_answer": benchmark_row["answer"],
            f"{method}_mode": run_mode,
            f"{method}_config": config,
            f"{method}_correct": expected_correct,
            f"{method}_answer": emitted_answer,
            f"{method}_completion": emitted_completion,
            f"{method}_completion_token_ids": "[42,151643]",
            f"{method}_completion_has_eos": True,
            f"{method}_time_s": p95,
            f"{method}_c_int": mean_c_int,
            f"{method}_c_tok": 2,
            f"{method}_c_step": 1,
            f"{method}_n_forward": 1,
            f"{method}_prefill_flops": mean_c_int,
            f"{method}_decode_flops": 0,
            f"{method}_total_flops": mean_c_int,
            f"{method}_pass_type": "fast" if run_mode == "adaptive" else "single",
            **_common_metadata(),
        }
        if method == "asmc":
            if run_mode == "adaptive":
                protocol_payload = ASMC_ADAPTIVE_PROTOCOL_PAYLOAD
                protocol_sha = ASMC_ADAPTIVE_PROTOCOL_SHA
            else:
                protocol_payload = ASMC_PROTOCOL_PAYLOAD
                protocol_sha = ASMC_PROTOCOL_SHA
            row.update(
                {
                    **_asmc_metadata(run_mode),
                    "asmc_protocol": "cache-coherent-asmc-corrected-v1",
                    "asmc_protocol_payload": protocol_payload,
                    "asmc_protocol_sha256": protocol_sha,
                    "asmc_budget_exhausted": False,
                    "asmc_budget_exhausted_at_token": "",
                    "asmc_stop_reason": "max_len",
                }
            )
        elif method == "greedy":
            row["greedy_protocol"] = "deterministic-greedy-decoding-v2"
        elif method == "naive":
            row["naive_protocol"] = "single-temperature-sample-v2"
        elif method == "bestofn":
            row.update(
                {
                    "bestofn_protocol": (
                        "independent-generation-unconditional-length-normalized-"
                        "logprob-argmax-v3"
                    ),
                    "bestofn_n": 2,
                    "bestofn_temperature": 0.25,
                    "bestofn_chunk_size": 8,
                }
            )
        elif method == "mcmc":
            row.update(
                {
                    "mcmc_protocol": "completion-only-eos-mcmc-power-sampling-v4",
                    "mcmc_steps": 2,
                    "mcmc_blocks": 8,
                    "mcmc_temperature": 0.25,
                }
            )
        method_identity = _build_method_rng_identity(
            row,
            method=method,
            config=config,
            location=f"fixture:{problem_idx + 2}",
        )
        rng_metadata = _derive_method_rng_metadata(
            row["seed"], problem_idx, method, method_identity
        )
        row.update(
            {
                "rng_protocol": RNG_PROTOCOL,
                f"{method}_rng_protocol": rng_metadata["protocol"],
                f"{method}_rng_seed": rng_metadata["seed"],
                f"{method}_rng_key_sha256": rng_metadata["key_sha256"],
                f"{method}_rng_key_payload": rng_metadata["key_payload"],
            }
        )
        rows.append(row)
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = audit_csvs(
        [raw_path],
        method=method,
        config=config,
        run_mode=run_mode,
        require_provenance=True,
    )
    summary_path = root / f"{series}.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _write_real_selection(root):
    # Build one shared raw CSV for fixed ASMC and all four baselines.  This
    # keeps the real raw->audit->selector derivation test while avoiding five
    # copies of the large MATH500 question/provenance columns on disk.
    combined_specs = [
        ("asmc", "asmc", ASMC_TEST_CONFIG, 0.75, 900000.0, 4.0),
        (
            "bestofn",
            "bestofn",
            "n2_temp0.25_chunk8_lengthnorm",
            0.73,
            800000.0,
            4.0,
        ),
        ("greedy", "greedy", "greedy", 0.70, 1100000.0, 3.0),
        (
            "mcmc",
            "mcmc",
            "steps2_blocks8_temp0.25",
            0.72,
            950000.0,
            4.0,
        ),
        ("naive", "naive", "temp0.25", 0.70, 500000.0, 3.0),
    ]
    combined_rows = [{} for _ in range(500)]
    for series, method, config, accuracy, mean_c_int, p95 in combined_specs:
        transient_summary = _write_audited_summary(
            root, method, config, accuracy, mean_c_int, p95
        )
        transient_raw = root / f"{series}.csv"
        with transient_raw.open(encoding="utf-8", newline="") as handle:
            method_rows = list(csv.DictReader(handle))
        if len(method_rows) != 500:
            raise AssertionError("fixture raw CSV must contain 500 rows")
        for problem_idx, method_row in enumerate(method_rows):
            merged = combined_rows[problem_idx]
            for field, value in method_row.items():
                if field in merged and merged[field] != value:
                    raise AssertionError(
                        f"fixture common field mismatch for {field!r}"
                    )
                merged[field] = value
        transient_raw.unlink()
        transient_summary.unlink()

    combined_raw = root / "combined-fixed-and-baselines.csv"
    with combined_raw.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    audit_paths = []
    for series, method, config, _accuracy, _mean_c_int, _p95 in combined_specs:
        summary = audit_csvs(
            [combined_raw],
            method=method,
            config=config,
            run_mode="fixed" if method == "asmc" else "single",
            require_provenance=True,
        )
        summary_path = root / f"{series}.summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        audit_paths.append(summary_path)

    audit_asmc_adaptive = _write_audited_summary(
        root,
        "asmc",
        ASMC_ADAPTIVE_TEST_CONFIG,
        0.76,
        850000.0,
        4.0,
        run_mode="adaptive",
    )
    audit_asmc, audit_bestofn, audit_greedy, audit_mcmc, audit_naive = audit_paths
    report = select_compute_matched(
        load_summary_jsons(
            [
                audit_asmc,
                audit_asmc_adaptive,
                audit_bestofn,
                audit_greedy,
                audit_mcmc,
                audit_naive,
            ]
        ),
        [1000000.0],
        require_provenance=True,
    )
    selection = root / "selection.json"
    selection.write_text(json.dumps(report), encoding="utf-8")
    return selection, report


class TableRendererTest(unittest.TestCase):
    def test_rejects_unverified_selection_by_default(self):
        report = _report()
        report["publication_provenance_required"] = False
        with self.assertRaisesRegex(RenderError, "--require-provenance"):
            validate_selection_report(report)

    def test_rejects_missing_method_and_cap_violation(self):
        report = _report()
        report["budgets"][0]["selections"].pop()
        with self.assertRaisesRegex(RenderError, "coverage mismatch"):
            validate_selection_report(report)

        report = _report()
        report["budgets"][0]["selections"][0]["mean_c_int"] = 1030000.0
        with self.assertRaisesRegex(RenderError, "exceeds the declared cap"):
            validate_selection_report(report)

        report = _report()
        report["budgets"][0]["selections"][0]["mean_c_int"] = 0.0
        with self.assertRaisesRegex(RenderError, "mean_c_int must be positive"):
            validate_selection_report(report)

        report = _report()
        report["budgets"][0]["cap_c_int"] = 1_500_000.0
        with self.assertRaisesRegex(RenderError, "budget_tolerance"):
            validate_selection_report(report)

        report = _report()
        report["n_problems"] = 10
        report["expected_problem_count"] = 10
        report["budgets"][0]["selections"][0]["n_problems"] = 10
        report["budgets"][0]["selections"][1]["n_problems"] = 10
        with self.assertRaisesRegex(RenderError, "cover 500 problems"):
            validate_selection_report(report)

        with self.assertRaisesRegex(RenderError, "exactly 500"):
            validate_selection_report(_report(), expected_problem_count=10)

    def test_output_paths_cannot_overwrite_dependencies_or_atomic_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.csv"
            raw_path.write_text("do-not-overwrite\n", encoding="utf-8")
            report = _report()
            report["input_summaries"][0]["source_artifacts"][0]["path"] = str(
                raw_path
            )
            selection = root / "selection.json"
            selection.write_text(json.dumps(report), encoding="utf-8")
            original_raw = raw_path.read_bytes()
            with self.assertRaisesRegex(RenderError, "must not overwrite"):
                render_outputs(
                    selection,
                    markdown_out=raw_path,
                    manifest_out=root / "manifest.json",
                )
            self.assertEqual(raw_path.read_bytes(), original_raw)

            with self.assertRaisesRegex(RenderError, "atomic temporary paths"):
                render_outputs(
                    selection,
                    markdown_out=root / "table.tmp",
                    manifest_out=root / "table",
                )
            self.assertFalse((root / "table.tmp").exists())
            self.assertFalse((root / "table").exists())

    def test_writes_tables_and_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, _ = _write_real_selection(root)
            latex = root / "table.tex"
            markdown = root / "table.md"
            manifest_path = root / "table.manifest.json"

            with patch(
                "analysis.render_compute_table._git_state",
                return_value={"git_commit": "a" * 40, "dirty": False},
            ):
                manifest = render_outputs(
                    selection,
                    latex_out=latex,
                    markdown_out=markdown,
                    manifest_out=manifest_path,
                )

            self.assertIn(
                _latex_config := ASMC_TEST_CONFIG.replace("_", r"\_"),
                latex.read_text(encoding="utf-8"),
            )
            self.assertIn("75.0", markdown.read_text(encoding="utf-8"))
            self.assertIn("--", markdown.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_rows"], 6)
            self.assertTrue(manifest["selection_derivation_verified"])
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["input"]["path"], str(selection))
            self.assertEqual(len(on_disk["input"]["sha256"]), 64)
            self.assertEqual(set(on_disk["outputs"]), {str(latex), str(markdown)})

    def test_rejects_a_tampered_selection_that_cannot_be_rederived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, report = _write_real_selection(root)
            report["budgets"][0]["selections"][0]["accuracy"] = 0.76
            selection.write_text(json.dumps(report), encoding="utf-8")

            with patch(
                "analysis.render_compute_table._git_state",
                return_value={"git_commit": "a" * 40, "dirty": False},
            ):
                with self.assertRaisesRegex(RenderError, "does not equal"):
                    render_outputs(
                        selection,
                        latex_out=root / "table.tex",
                        manifest_out=root / "table.manifest.json",
                    )

    def test_selector_rejects_a_self_certified_forged_audit_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = _write_audited_summary(
                root, "asmc", ASMC_TEST_CONFIG, 0.75, 900000.0, 4.0
            )
            forged = json.loads(summary_path.read_text(encoding="utf-8"))
            forged["accuracy"] = 1.0
            forged["provenance_complete"] = True
            summary_path.write_text(json.dumps(forged), encoding="utf-8")

            with self.assertRaisesRegex(
                SelectionError, "does not equal the result re-derived"
            ):
                select_compute_matched(
                    load_summary_jsons([summary_path]),
                    [1000000.0],
                    require_provenance=True,
                )


if __name__ == "__main__":
    unittest.main()
