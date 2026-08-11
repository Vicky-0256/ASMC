#!/usr/bin/env python3
"""Render a compute-matched selection report into reviewable paper tables.

The renderer deliberately accepts only the structured JSON emitted by
``select_compute_matched.py``.  In publication mode it re-runs the complete
raw-CSV -> audit-summary -> deterministic-selection derivation and rejects any
hash or content mismatch.  The accompanying manifest records the exact input
and output hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


class RenderError(ValueError):
    """Raised when a selection report cannot support a paper table."""


ASMC_PUBLICATION_PROFILE = "corrected-paper-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository: Path) -> dict[str, object]:
    """Return observed Git provenance without inventing a revision."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"git_commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "dirty": None}


def _finite_number(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"{location} must be numeric, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RenderError(f"{location} must be finite, got {value!r}")
    return parsed


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderError(f"{location} must be a non-empty string")
    return value.strip()


def load_selection_report(path: str | Path) -> dict[str, object]:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except OSError as exc:
        raise RenderError(f"cannot read selection report {input_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"invalid selection JSON {input_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise RenderError("selection JSON root must be an object")
    return report


def _verify_selection_derivation(report: Mapping[str, object]) -> None:
    """Re-run the deterministic selector over the hash-checked audit inputs."""

    try:
        from .select_compute_matched import (
            SelectionError,
            load_summary_jsons,
            select_compute_matched,
        )
    except ImportError:  # Direct execution from the repository root.
        from select_compute_matched import (  # type: ignore
            SelectionError,
            load_summary_jsons,
            select_compute_matched,
        )

    paths = [artifact["path"] for artifact in report["input_summaries"]]
    budgets = [entry["budget_c_int"] for entry in report["budgets"]]
    try:
        summaries = load_summary_jsons(paths)
        recomputed = select_compute_matched(
            summaries,
            budgets,
            tolerance=report["budget_tolerance"],
            require_provenance=True,
            expected_problem_count=report["expected_problem_count"],
        )
    except (SelectionError, OSError, TypeError, ValueError) as exc:
        raise RenderError(
            f"selection report cannot be re-derived from its audit summaries: {exc}"
        ) from exc
    if recomputed != dict(report):
        raise RenderError(
            "selection report does not equal the deterministic result re-derived "
            "from its audit summaries"
        )


def validate_selection_report(
    report: Mapping[str, object],
    *,
    require_provenance: bool = True,
    expected_problem_count: int = 500,
) -> list[dict[str, object]]:
    """Validate and flatten one deterministic row per budget/method."""

    if report.get("schema_version") != 1:
        raise RenderError("selection report schema_version must be 1")
    n_candidates = report.get("n_candidates")
    if (
        isinstance(n_candidates, bool)
        or not isinstance(n_candidates, int)
        or n_candidates <= 0
    ):
        raise RenderError("selection report n_candidates must be a positive integer")
    methods_value = report.get("methods")
    if not isinstance(methods_value, list) or not methods_value:
        raise RenderError("selection report methods must be a non-empty list")
    methods = [
        _nonempty_string(value, location=f"methods[{index}]")
        for index, value in enumerate(methods_value)
    ]
    if len(set(methods)) != len(methods):
        raise RenderError("selection report methods must be unique")

    if (
        isinstance(expected_problem_count, bool)
        or not isinstance(expected_problem_count, int)
        or expected_problem_count <= 0
    ):
        raise RenderError("expected_problem_count must be a positive integer")
    if require_provenance and expected_problem_count != 500:
        raise RenderError(
            "publication rendering requires exactly 500 MATH500 problems"
        )
    n_problems = report.get("n_problems")
    if isinstance(n_problems, bool) or not isinstance(n_problems, int) or n_problems <= 0:
        raise RenderError("selection report n_problems must be a positive integer")
    if n_problems != expected_problem_count:
        raise RenderError(
            f"selection report must cover {expected_problem_count} problems, "
            f"got {n_problems}"
        )
    if report.get("expected_problem_count") != expected_problem_count:
        raise RenderError(
            "selection report expected_problem_count does not match the renderer gate"
        )
    tolerance = _finite_number(
        report.get("budget_tolerance"), location="budget_tolerance"
    )
    if tolerance < 1.0:
        raise RenderError("budget_tolerance must be at least 1.0")
    if require_provenance and not math.isclose(
        tolerance, 1.02, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RenderError("publication budget_tolerance must equal 1.02")

    candidate_pairs: set[tuple[str, str]] = set()
    candidate_counts: Counter[str] = Counter()
    series_identities: dict[str, tuple[str, str]] = {}
    baseline_c0: float | None = None
    budget_multipliers: list[object] | None = None
    if require_provenance:
        if report.get("publication_provenance_required") is not True:
            raise RenderError(
                "selection report was not produced with --require-provenance"
            )
        if report.get("comparability_metadata_complete") is not True:
            raise RenderError("selection report comparability metadata is incomplete")
        if report.get("raw_audit_derivation_verified") is not True:
            raise RenderError(
                "selection report did not verify raw CSV -> audit derivation"
            )
        if report.get("asmc_publication_profile") != ASMC_PUBLICATION_PROFILE:
            raise RenderError(
                "selection report does not enforce the corrected ASMC "
                "publication profile"
            )
        required_series = report.get("required_publication_series")
        expected_series = [
            "asmc",
            "asmc-adaptive",
            "bestofn",
            "greedy",
            "mcmc",
            "naive",
        ]
        if required_series != expected_series:
            raise RenderError(
                "selection report does not declare the complete publication roster"
            )
        if not set(expected_series).issubset(methods):
            raise RenderError(
                "selection report is missing a required publication series"
            )
        input_summaries = report.get("input_summaries")
        if not isinstance(input_summaries, list) or len(input_summaries) != n_candidates:
            raise RenderError(
                "publication selection must identify every input audit summary"
            )
        for index, artifact in enumerate(input_summaries):
            location = f"input_summaries[{index}]"
            if not isinstance(artifact, Mapping):
                raise RenderError(f"{location} must be an object")
            _nonempty_string(artifact.get("path"), location=f"{location}.path")
            sha256 = artifact.get("sha256")
            if not isinstance(sha256, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", sha256
            ):
                raise RenderError(f"{location}.sha256 must be 64 hexadecimal digits")
            if artifact.get("audit_schema_version") != 1:
                raise RenderError(f"{location}.audit_schema_version must be 1")
            artifact_method = _nonempty_string(
                artifact.get("method"), location=f"{location}.method"
            )
            metric_method = _nonempty_string(
                artifact.get("metric_method"),
                location=f"{location}.metric_method",
            ).lower()
            run_mode = _nonempty_string(
                artifact.get("run_mode"), location=f"{location}.run_mode"
            ).lower()
            expected_series = (
                "asmc-adaptive"
                if metric_method == "asmc" and run_mode == "adaptive"
                else metric_method
            )
            if metric_method == "asmc" and run_mode not in {"fixed", "adaptive"}:
                raise RenderError(f"{location} has an invalid ASMC run_mode")
            if metric_method != "asmc" and run_mode != "single":
                raise RenderError(f"{location} has an invalid baseline run_mode")
            if artifact_method != expected_series:
                raise RenderError(
                    f"{location}.method does not match metric_method/run_mode"
                )
            previous_identity = series_identities.setdefault(
                artifact_method, (metric_method, run_mode)
            )
            if previous_identity != (metric_method, run_mode):
                raise RenderError(
                    f"{location} mixes protocol identities within one series"
                )
            artifact_config = _nonempty_string(
                artifact.get("config"), location=f"{location}.config"
            )
            source_artifacts = artifact.get("source_artifacts")
            if not isinstance(source_artifacts, list) or not source_artifacts:
                raise RenderError(
                    f"{location}.source_artifacts must identify raw CSV files"
                )
            for source_index, source in enumerate(source_artifacts):
                source_location = f"{location}.source_artifacts[{source_index}]"
                if not isinstance(source, Mapping):
                    raise RenderError(f"{source_location} must be an object")
                _nonempty_string(
                    source.get("path"), location=f"{source_location}.path"
                )
                source_sha = source.get("sha256")
                if not isinstance(source_sha, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", source_sha
                ):
                    raise RenderError(
                        f"{source_location}.sha256 must be 64 hexadecimal digits"
                    )
            if artifact_method not in methods:
                raise RenderError(
                    f"{location}.method is not declared by the selection report"
                )
            pair = (artifact_method, artifact_config)
            if pair in candidate_pairs:
                raise RenderError(
                    f"duplicate input audit summary for method/config {pair!r}"
                )
            candidate_pairs.add(pair)
            candidate_counts[artifact_method] += 1
        if set(candidate_counts) != set(methods):
            raise RenderError(
                "input audit summaries do not cover every declared method"
            )
        baseline = report.get("budget_baseline")
        if not isinstance(baseline, Mapping):
            raise RenderError("publication selection must identify its C0 baseline")
        if baseline.get("method") != "naive":
            raise RenderError("publication C0 baseline must be the naive method")
        baseline_config = _nonempty_string(
            baseline.get("config"), location="budget_baseline.config"
        )
        if ("naive", baseline_config) not in candidate_pairs:
            raise RenderError("publication C0 baseline is not an audited candidate")
        baseline_c0 = _finite_number(
            baseline.get("mean_c_int"), location="budget_baseline.mean_c_int"
        )
        if baseline_c0 <= 0:
            raise RenderError("budget_baseline.mean_c_int must be positive")
        multipliers_value = report.get("budget_multipliers")
        if not isinstance(multipliers_value, list):
            raise RenderError("budget_multipliers must be a list")
        budget_multipliers = multipliers_value

    budgets_value = report.get("budgets")
    if not isinstance(budgets_value, list) or not budgets_value:
        raise RenderError("selection report budgets must be a non-empty list")
    if require_provenance and len(budget_multipliers) != len(budgets_value):
        raise RenderError("budget_multipliers must align with budgets")

    flattened: list[dict[str, object]] = []
    seen_budgets: set[float] = set()
    previous_budget = float("-inf")
    for budget_index, budget_report in enumerate(budgets_value):
        location = f"budgets[{budget_index}]"
        if not isinstance(budget_report, Mapping):
            raise RenderError(f"{location} must be an object")
        budget = _finite_number(
            budget_report.get("budget_c_int"), location=f"{location}.budget_c_int"
        )
        if require_provenance:
            multiplier = budget_multipliers[budget_index]
            if (
                isinstance(multiplier, bool)
                or not isinstance(multiplier, int)
                or multiplier not in {2, 4, 8, 16, 32, 64, 128}
            ):
                raise RenderError(f"{location} has an invalid C0 multiplier")
            expected_budget = baseline_c0 * multiplier
            if not math.isclose(
                budget, expected_budget, rel_tol=1e-12, abs_tol=1e-6
            ):
                raise RenderError(
                    f"{location}.budget_c_int does not equal {multiplier} x C0"
                )
        cap = _finite_number(
            budget_report.get("cap_c_int"), location=f"{location}.cap_c_int"
        )
        expected_cap = budget * tolerance
        if budget <= 0 or cap < budget:
            raise RenderError(f"{location} has an invalid budget/cap")
        if not math.isclose(cap, expected_cap, rel_tol=1e-12, abs_tol=1e-9):
            raise RenderError(
                f"{location}.cap_c_int must equal budget_tolerance * budget_c_int "
                f"({expected_cap:g}), got {cap:g}"
            )
        if budget in seen_budgets or budget <= previous_budget:
            raise RenderError("budgets must be unique and strictly increasing")
        seen_budgets.add(budget)
        previous_budget = budget

        selections = budget_report.get("selections")
        if not isinstance(selections, list):
            raise RenderError(f"{location}.selections must be a list")
        by_method: dict[str, Mapping[str, object]] = {}
        for selection_index, selection in enumerate(selections):
            selection_location = f"{location}.selections[{selection_index}]"
            if not isinstance(selection, Mapping):
                raise RenderError(f"{selection_location} must be an object")
            method = _nonempty_string(
                selection.get("method"), location=f"{selection_location}.method"
            )
            if method in by_method:
                raise RenderError(f"{location} contains duplicate method {method!r}")
            by_method[method] = selection

        if set(by_method) != set(methods):
            missing = sorted(set(methods) - set(by_method))
            extra = sorted(set(by_method) - set(methods))
            raise RenderError(
                f"{location} method coverage mismatch; missing={missing}, extra={extra}"
            )

        for method in methods:
            selection = by_method[method]
            status = selection.get("status")
            if status not in {"selected", "no_eligible_config"}:
                raise RenderError(
                    f"{location}.{method}.status is invalid: {status!r}"
                )
            row: dict[str, object] = {
                "budget_c_int": budget,
                "cap_c_int": cap,
                "method": method,
                "status": status,
                "n_problems": n_problems,
            }
            if require_provenance:
                metric_method, run_mode = series_identities[method]
                if selection.get("metric_method") != metric_method:
                    raise RenderError(
                        f"{location}.{method}.metric_method does not match its "
                        "audited series"
                    )
                if selection.get("run_mode") != run_mode:
                    raise RenderError(
                        f"{location}.{method}.run_mode does not match its "
                        "audited series"
                    )
            eligible_count = selection.get("eligible_config_count")
            if (
                isinstance(eligible_count, bool)
                or not isinstance(eligible_count, int)
                or eligible_count < 0
            ):
                raise RenderError(
                    f"{location}.{method}.eligible_config_count must be a "
                    "non-negative integer"
                )
            if status == "selected":
                if eligible_count < 1:
                    raise RenderError(
                        f"{location}.{method} selected a configuration but "
                        "eligible_config_count is zero"
                    )
                if require_provenance and eligible_count > candidate_counts[method]:
                    raise RenderError(
                        f"{location}.{method}.eligible_config_count exceeds "
                        "the audited candidate count"
                    )
                config = _nonempty_string(
                    selection.get("config"), location=f"{location}.{method}.config"
                )
                if require_provenance and (method, config) not in candidate_pairs:
                    raise RenderError(
                        f"{location}.{method} selected an unaudited configuration"
                    )
                accuracy = _finite_number(
                    selection.get("accuracy"),
                    location=f"{location}.{method}.accuracy",
                )
                mean_c_int = _finite_number(
                    selection.get("mean_c_int"),
                    location=f"{location}.{method}.mean_c_int",
                )
                time_p50_s = _finite_number(
                    selection.get("time_p50_s"),
                    location=f"{location}.{method}.time_p50_s",
                )
                time_p95_s = _finite_number(
                    selection.get("time_p95_s"),
                    location=f"{location}.{method}.time_p95_s",
                )
                selected_n = selection.get("n_problems")
                if selected_n != n_problems:
                    raise RenderError(
                        f"{location}.{method}.n_problems does not match the report"
                    )
                if not 0.0 <= accuracy <= 1.0:
                    raise RenderError(f"{location}.{method}.accuracy is outside [0, 1]")
                if mean_c_int <= 0:
                    raise RenderError(
                        f"{location}.{method}.mean_c_int must be positive"
                    )
                if min(time_p50_s, time_p95_s) < 0:
                    raise RenderError(f"{location}.{method} contains a negative metric")
                if mean_c_int > cap:
                    raise RenderError(f"{location}.{method} exceeds the declared cap")
                if time_p95_s < time_p50_s:
                    raise RenderError(f"{location}.{method} has p95 below p50")
                row.update(
                    {
                        "config": config,
                        "accuracy": accuracy,
                        "mean_c_int": mean_c_int,
                        "time_p50_s": time_p50_s,
                        "time_p95_s": time_p95_s,
                    }
                )
            else:
                if eligible_count != 0:
                    raise RenderError(
                        f"{location}.{method} has no eligible configuration but "
                        "eligible_config_count is not zero"
                    )
                for field in (
                    "config",
                    "accuracy",
                    "mean_c_int",
                    "time_p50_s",
                    "time_p95_s",
                ):
                    if selection.get(field) is not None:
                        raise RenderError(
                            f"{location}.{method}.{field} must be null when no config is eligible"
                        )
                    row[field] = None
            instance_cap = selection.get("per_instance_c_int_cap")
            if require_provenance and method == "asmc-adaptive" and status == "selected":
                parsed_instance_cap = _finite_number(
                    instance_cap,
                    location=f"{location}.{method}.per_instance_c_int_cap",
                )
                if not math.isclose(
                    parsed_instance_cap,
                    budget,
                    rel_tol=1e-12,
                    abs_tol=1e-6,
                ):
                    raise RenderError(
                        f"{location}.{method}.per_instance_c_int_cap must equal "
                        "the target budget"
                    )
                row["per_instance_c_int_cap"] = parsed_instance_cap
            else:
                if instance_cap is not None:
                    raise RenderError(
                        f"{location}.{method}.per_instance_c_int_cap must be null"
                    )
                row["per_instance_c_int_cap"] = None
            flattened.append(row)
    return flattened


_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value: object) -> str:
    return "".join(_LATEX_REPLACEMENTS.get(char, char) for char in str(value))


def _markdown_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|")


def _display_metrics(row: Mapping[str, object]) -> tuple[str, str, str, str, str]:
    if row["status"] != "selected":
        return ("--", "--", "--", "--", "--")
    return (
        str(row["config"]),
        f"{100.0 * float(row['accuracy']):.1f}",
        f"{float(row['mean_c_int']) / 1_000_000.0:.3f}",
        f"{float(row['time_p50_s']):.1f}",
        f"{float(row['time_p95_s']):.1f}",
    )


def render_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Budget C_int | Method | Configuration | Accuracy (%) | Mean C_int (M) | p50 (s) | p95 (s) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        config, accuracy, mean_c_int, p50, p95 = _display_metrics(row)
        lines.append(
            "| {budget:g} | {method} | {config} | {accuracy} | {mean_c_int} | {p50} | {p95} |".format(
                budget=float(row["budget_c_int"]),
                method=_markdown_escape(row["method"]),
                config=_markdown_escape(config),
                accuracy=accuracy,
                mean_c_int=mean_c_int,
                p50=p50,
                p95=p95,
            )
        )
    return "\n".join(lines) + "\n"


def render_latex(
    rows: Sequence[Mapping[str, object]],
    *,
    caption: str = "Compute-matched MATH500 results from audited run manifests.",
    label: str = "tab:compute-matched",
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9:./-]+", label):
        raise RenderError(
            "LaTeX label may contain only letters, digits, colon, dot, slash, and hyphen"
        )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{rllrrrr}",
        r"\toprule",
        r"Budget $C_{\mathrm{int}}$ & Method & Configuration & Accuracy (\%) & Mean $C_{\mathrm{int}}$ (M) & p50 (s) & p95 (s) \\",
        r"\midrule",
    ]
    previous_budget: float | None = None
    for row in rows:
        budget = float(row["budget_c_int"])
        if previous_budget is not None and budget != previous_budget:
            lines.append(r"\midrule")
        previous_budget = budget
        config, accuracy, mean_c_int, p50, p95 = _display_metrics(row)
        lines.append(
            "{budget:g} & {method} & {config} & {accuracy} & {mean_c_int} & {p50} & {p95} \\\\".format(
                budget=budget,
                method=_latex_escape(row["method"]),
                config=_latex_escape(config),
                accuracy=accuracy,
                mean_c_int=mean_c_int,
                p50=p50,
                p95=p95,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary_path, path)


def render_outputs(
    report_path: str | Path,
    *,
    latex_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
    manifest_out: str | Path,
    require_provenance: bool = True,
    expected_problem_count: int = 500,
    caption: str = "Compute-matched MATH500 results from audited run manifests.",
    label: str = "tab:compute-matched",
) -> dict[str, object]:
    input_path = Path(report_path)
    outputs = [Path(path) for path in (latex_out, markdown_out) if path is not None]
    if not outputs:
        raise RenderError("at least one of latex_out or markdown_out is required")
    manifest_path = Path(manifest_out)
    input_resolved = input_path.resolve()
    write_paths = [manifest_path.resolve()] + [path.resolve() for path in outputs]
    atomic_write_paths = []
    for path in write_paths:
        atomic_write_paths.extend([path, path.with_suffix(path.suffix + ".tmp")])
    if (
        len(set(atomic_write_paths)) != len(atomic_write_paths)
        or input_resolved in atomic_write_paths
    ):
        raise RenderError(
            "selection input, outputs, manifest, and all atomic temporary paths "
            "must be distinct"
        )

    report = load_selection_report(input_path)
    repository = Path(__file__).resolve().parents[1]
    dependency_paths = {
        input_resolved,
        Path(__file__).resolve(),
        (repository / "analysis" / "select_compute_matched.py").resolve(),
        (repository / "analysis" / "result_audit.py").resolve(),
        (repository / "data" / "MATH500.json").resolve(),
    }
    input_summaries = report.get("input_summaries", [])
    if isinstance(input_summaries, list):
        for artifact in input_summaries:
            if not isinstance(artifact, Mapping):
                continue
            path_value = artifact.get("path")
            if isinstance(path_value, str) and path_value.strip():
                dependency_paths.add(Path(path_value).resolve())
            source_artifacts = artifact.get("source_artifacts", [])
            if isinstance(source_artifacts, list):
                for source in source_artifacts:
                    if isinstance(source, Mapping):
                        source_path = source.get("path")
                        if isinstance(source_path, str) and source_path.strip():
                            dependency_paths.add(Path(source_path).resolve())
    overlapping_paths = dependency_paths.intersection(atomic_write_paths)
    if overlapping_paths:
        raise RenderError(
            "a renderer output or temporary path must not overwrite a selection, "
            "audit summary, raw CSV, benchmark, or generator dependency: "
            + ", ".join(str(path) for path in sorted(overlapping_paths))
        )
    rows = validate_selection_report(
        report,
        require_provenance=require_provenance,
        expected_problem_count=expected_problem_count,
    )
    generator_code = _git_state(repository)
    if require_provenance:
        if not generator_code.get("git_commit"):
            raise RenderError("publication rendering requires a Git commit")
        if generator_code.get("dirty") is not False:
            raise RenderError("publication rendering requires a clean Git worktree")
        for index, artifact in enumerate(report["input_summaries"]):
            artifact_path = Path(str(artifact["path"]))
            if not artifact_path.is_file():
                raise RenderError(
                    f"input_summaries[{index}] does not exist: {artifact_path}"
                )
            observed_sha = _sha256(artifact_path)
            if observed_sha.lower() != str(artifact["sha256"]).lower():
                raise RenderError(
                    f"input_summaries[{index}] SHA-256 does not match: {artifact_path}"
                )
        _verify_selection_derivation(report)
    dependency_hashes = (
        {
            path: _sha256(path)
            for path in dependency_paths
            if path.is_file()
        }
        if require_provenance
        else {}
    )
    if latex_out is not None:
        _atomic_write_text(
            Path(latex_out), render_latex(rows, caption=caption, label=label)
        )
    if markdown_out is not None:
        _atomic_write_text(Path(markdown_out), render_markdown(rows))

    output_metadata = {
        str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in outputs
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": "analysis/render_compute_table.py",
        "generator_code": generator_code,
        "generator_source": {
            "path": "analysis/render_compute_table.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "publication_provenance_required": require_provenance,
        "selection_derivation_verified": require_provenance,
        "raw_audit_derivation_verified": require_provenance,
        "asmc_publication_profile": report.get("asmc_publication_profile"),
        "expected_problem_count": expected_problem_count,
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
        },
        "outputs": output_metadata,
        "rounding": {
            "accuracy_percent_decimals": 1,
            "mean_c_int_millions_decimals": 3,
            "latency_seconds_decimals": 1,
        },
        "n_rows": len(rows),
        "selection_common_invariant_metadata": report.get(
            "common_invariant_metadata", {}
        ),
        "input_audit_summaries": report.get("input_summaries", []),
    }
    _atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if not manifest_path.is_file():
        raise RenderError(f"renderer manifest was not created: {manifest_path}")
    for path_text, metadata in output_metadata.items():
        path = Path(path_text)
        if not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise RenderError(f"renderer output changed while writing: {path}")
    for path, expected_sha256 in dependency_hashes.items():
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise RenderError(
                f"publication input changed while writing outputs: {path}"
            )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic LaTeX/Markdown tables from one audited "
            "compute-matched selection JSON."
        )
    )
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("--latex-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help="diagnostic only: render a report not produced by the publication gate",
    )
    parser.add_argument(
        "--caption",
        default="Compute-matched MATH500 results from audited run manifests.",
    )
    parser.add_argument(
        "--expected-problems",
        type=int,
        default=500,
        help="required benchmark size (default: 500)",
    )
    parser.add_argument("--label", default="tab:compute-matched")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = render_outputs(
            args.selection_json,
            latex_out=args.latex_out,
            markdown_out=args.markdown_out,
            manifest_out=args.manifest_out,
            require_provenance=not args.allow_incomplete_provenance,
            expected_problem_count=args.expected_problems,
            caption=args.caption,
            label=args.label,
        )
    except (RenderError, OSError) as exc:
        print(f"table render failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
