#!/usr/bin/env python3
"""Select compute-matched configurations from audited result summaries.

Inputs are the JSON objects emitted by ``analysis/result_audit.py``.  For each
absolute C_int budget and method, the selector considers configurations whose
mean C_int is at most ``budget * tolerance``.  It then applies the paper's
deterministic ranking rule: highest accuracy, lowest mean C_int, lowest p95
latency, and finally lexicographically smallest configuration name.

The selector validates all candidates before doing any selection.  In
particular, it will not compare partial benchmarks, silently overwrite a
duplicate method/configuration pair, or rank non-finite measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


REQUIRED_FIELDS = (
    "schema_version",
    "method",
    "config",
    "run_mode",
    "accuracy",
    "mean_c_int",
    "time_p50_s",
    "time_p95_s",
    "n_problems",
)

COMPARABILITY_FIELDS = (
    "model_id",
    "model_revision",
    "dtype",
    "attn_implementation",
    "trust_remote_code",
    "dataset_name",
    "dataset_sha256",
    "code_git_commit",
    "code_git_dirty",
    "python_version",
    "pytorch_version",
    "transformers_version",
    "cuda_runtime",
    "gpu_name",
    "nvidia_driver_version",
    "flash_attn_version",
    "cot",
    "max_tokens",
    "temperature",
    "seed",
    "compute_schema",
    "timing_schema",
)

_ARTIFACT_KEY = "_audit_summary_artifact"
ASMC_PUBLICATION_PROFILE = "corrected-paper-v1"
PAPER_MODEL_ID = "Qwen/Qwen2.5-Math-7B"
PAPER_DATASET_SHA256 = (
    "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
)
GENERATION_METHODS = frozenset(
    {"greedy", "naive", "std", "majority", "mcmc", "bestofn"}
)
PUBLICATION_BASELINE_PROTOCOLS = {
    "greedy": "deterministic-greedy-decoding-v2",
    "naive": "single-temperature-sample-v2",
    "std": "single-temperature-one-sample-v2",
    "mcmc": "completion-only-eos-mcmc-power-sampling-v4",
    "majority": "independent-sampling-unweighted-answer-majority-v2",
    "bestofn": (
        "independent-generation-unconditional-length-normalized-"
        "logprob-argmax-v3"
    ),
}


class SelectionError(ValueError):
    """Raised when summaries cannot support a valid compute-matched table."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Candidate:
    """Validated metrics for one method/configuration pair."""

    metric_method: str
    method: str
    run_mode: str
    config: str
    accuracy: float
    mean_c_int: float
    time_p50_s: float
    time_p95_s: float
    n_problems: int

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "metric_method": self.metric_method,
            "run_mode": self.run_mode,
            "config": self.config,
            "accuracy": self.accuracy,
            "mean_c_int": self.mean_c_int,
            "time_p50_s": self.time_p50_s,
            "time_p95_s": self.time_p95_s,
            "n_problems": self.n_problems,
        }


def _location(index: int, source: str | None) -> str:
    return source if source is not None else f"summary[{index}]"


def _require_label(
    summary: Mapping[str, object], field: str, *, location: str
) -> str:
    value = summary[field]
    if not isinstance(value, str) or not value.strip():
        raise SelectionError(f"{location}: {field} must be a non-empty string")
    return value.strip()


def _require_finite_number(
    summary: Mapping[str, object], field: str, *, location: str
) -> float:
    value = summary[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{location}: {field} must be numeric, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SelectionError(f"{location}: {field} must be finite, got {value!r}")
    return parsed


def _selection_method(metric_method: str, run_mode: str, *, location: str) -> str:
    """Map an audited metric prefix/protocol to one publication table series."""

    if metric_method == "asmc":
        if run_mode == "fixed":
            return "asmc"
        if run_mode == "adaptive":
            return "asmc-adaptive"
        raise SelectionError(
            f"{location}: ASMC run_mode must be 'fixed' or 'adaptive'"
        )
    if run_mode != "single":
        raise SelectionError(
            f"{location}: non-ASMC run_mode must be 'single', got {run_mode!r}"
        )
    if metric_method == "asmc-adaptive":
        raise SelectionError(
            f"{location}: metric method 'asmc-adaptive' is reserved for the "
            "adaptive ASMC publication series"
        )
    return metric_method


def _metadata_bool(metadata: Mapping[str, object], field: str, *, location: str) -> bool:
    if field not in metadata:
        raise SelectionError(f"{location}: missing publication metadata {field}")
    normalized = str(metadata[field]).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise SelectionError(
        f"{location}: publication metadata {field} must be boolean, "
        f"got {metadata[field]!r}"
    )


def _metadata_float(
    metadata: Mapping[str, object], field: str, *, location: str
) -> float:
    if field not in metadata:
        raise SelectionError(f"{location}: missing publication metadata {field}")
    try:
        value = float(str(metadata[field]).strip())
    except ValueError as exc:
        raise SelectionError(
            f"{location}: publication metadata {field} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise SelectionError(
            f"{location}: publication metadata {field} must be finite"
        )
    return value


def _validate_common_publication_profile(
    metadata: Mapping[str, object], *, location: str
) -> None:
    expected_strings = {
        "model_id": PAPER_MODEL_ID,
        "dataset_name": "MATH500",
        "dataset_sha256": PAPER_DATASET_SHA256,
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "compute_schema": "asmc-compute-v2",
        "timing_schema": "synchronized-end-to-end-wall-clock-v1",
    }
    for field, expected in expected_strings.items():
        observed = str(metadata.get(field, "")).strip()
        if observed != expected:
            raise SelectionError(
                f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
                f"{field}={expected!r}, got {observed!r}"
            )
    for field, expected in (
        ("temperature", 0.25),
        ("max_tokens", 3072.0),
        ("seed", 0.0),
    ):
        observed = _metadata_float(metadata, field, location=location)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise SelectionError(
                f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
                f"{field}={expected:g}, got {observed:g}"
            )
    if _metadata_bool(metadata, "trust_remote_code", location=location):
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
            "trust_remote_code=False"
        )
    if not _metadata_bool(metadata, "cot", location=location):
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires cot=True"
        )
    gpu_name = str(metadata.get("gpu_name", ""))
    if "A100" not in gpu_name or "80GB" not in gpu_name:
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires an A100 80GB "
            f"GPU identity, got {gpu_name!r}"
        )
    cuda_runtime = str(metadata.get("cuda_runtime", "")).strip()
    if not cuda_runtime.startswith("12."):
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires a resolved "
            f"CUDA 12.x runtime, got {cuda_runtime!r}"
        )


def _validate_asmc_publication_profile(
    summary: Mapping[str, object],
    candidate: Candidate,
    metadata: Mapping[str, object],
    *,
    location: str,
) -> None:
    """Reject legacy or protocol-changing ASMC candidates from paper tables."""

    if summary.get("legacy_aliases_allowed") is not False:
        raise SelectionError(
            f"{location}: publication selection forbids legacy metric aliases"
        )
    if candidate.metric_method != "asmc":
        return
    protocol = str(metadata.get("asmc_protocol", "")).strip()
    if protocol != "cache-coherent-asmc-corrected-v1":
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires the corrected "
            "ASMC protocol identifier"
        )
    required_values = {
        "asmc_vote_mode": "weighted_no_source",
        "asmc_backend": "batched",
        "asmc_anneal_schedule": "cosine",
    }
    for field, expected in required_values.items():
        observed = str(metadata.get(field, "")).strip().lower()
        if observed != expected:
            raise SelectionError(
                f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
                f"{field}={expected!r}, got {observed!r}"
            )
    for field, expected in (
        ("asmc_use_source_weight", False),
        ("asmc_legacy_stop_constraints", False),
        ("asmc_use_batched", True),
    ):
        if _metadata_bool(metadata, field, location=location) != expected:
            raise SelectionError(
                f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
                f"{field}={expected}"
            )

    alpha_star = _metadata_float(metadata, "asmc_alpha_star", location=location)
    if not math.isclose(alpha_star, 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
            f"asmc_alpha_star=4, got {alpha_star:g}"
        )

    n_particles = _metadata_float(metadata, "asmc_n_particles", location=location)
    fast_n = _metadata_float(metadata, "asmc_fast_n_particles", location=location)
    hard_n = _metadata_float(metadata, "asmc_hard_n_particles", location=location)
    if n_particles < 1 or not n_particles.is_integer():
        raise SelectionError(f"{location}: asmc_n_particles must be a positive integer")
    if fast_n != max(1.0, n_particles // 2) or hard_n != n_particles:
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires fast N=N/2 "
            "and hard N=N"
        )
    protocol_sha = str(metadata.get("asmc_protocol_sha256", "")).strip().lower()
    protocol_payload = str(metadata.get("asmc_protocol_payload", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_sha):
        raise SelectionError(f"{location}: invalid asmc_protocol_sha256")
    if hashlib.sha256(protocol_payload.encode("utf-8")).hexdigest() != protocol_sha:
        raise SelectionError(
            f"{location}: asmc_protocol_sha256 does not match its payload"
        )
    expected_prefix = (
        f"asmc-{candidate.run_mode}-n{int(n_particles)}-"
        "weighted_no_source-"
    )
    expected_config = expected_prefix + protocol_sha[:16]
    if candidate.config != expected_config:
        raise SelectionError(
            f"{location}: ASMC config must be the runner's content-addressed "
            f"identifier {expected_config!r}"
        )

    cap_text = str(metadata.get("asmc_c_int_cap", "")).strip().lower()
    if candidate.run_mode == "fixed":
        if cap_text != "none":
            raise SelectionError(
                f"{location}: fixed {ASMC_PUBLICATION_PROFILE} requires "
                "asmc_c_int_cap='none'"
            )
    else:
        try:
            cap = float(cap_text)
        except ValueError as exc:
            raise SelectionError(
                f"{location}: adaptive {ASMC_PUBLICATION_PROFILE} requires a "
                "positive finite asmc_c_int_cap"
            ) from exc
        if not math.isfinite(cap) or cap <= 0:
            raise SelectionError(
                f"{location}: adaptive {ASMC_PUBLICATION_PROFILE} requires a "
                "positive finite asmc_c_int_cap"
            )


def _validate_baseline_publication_profile(
    candidate: Candidate,
    metadata: Mapping[str, object],
    *,
    location: str,
) -> None:
    if candidate.metric_method == "asmc":
        return
    expected_protocol = PUBLICATION_BASELINE_PROTOCOLS.get(
        candidate.metric_method
    )
    if expected_protocol is None:
        raise SelectionError(
            f"{location}: method {candidate.metric_method!r} is not part of "
            f"{ASMC_PUBLICATION_PROFILE}"
        )
    protocol_field = f"{candidate.metric_method}_protocol"
    observed_protocol = str(metadata.get(protocol_field, "")).strip()
    if observed_protocol != expected_protocol:
        raise SelectionError(
            f"{location}: {ASMC_PUBLICATION_PROFILE} requires "
            f"{protocol_field}={expected_protocol!r}, got {observed_protocol!r}"
        )

    if candidate.metric_method in GENERATION_METHODS:
        try:
            from .result_audit import (
                AuditError,
                _publication_sampling_config,
            )
        except ImportError:  # Direct execution from the repository root.
            from result_audit import (  # type: ignore
                AuditError,
                _publication_sampling_config,
            )
        try:
            _publication_sampling_config(metadata, location=location)
        except AuditError as exc:
            raise SelectionError(str(exc)) from exc

    expected_config: str
    if candidate.metric_method == "greedy":
        expected_config = "greedy"
    elif candidate.metric_method == "naive":
        expected_config = "temp0.25"
    elif candidate.metric_method == "std":
        expected_config = "temp1"
    elif candidate.metric_method == "mcmc":
        steps = _metadata_float(metadata, "mcmc_steps", location=location)
        blocks = _metadata_float(metadata, "mcmc_blocks", location=location)
        max_tokens = _metadata_float(metadata, "max_tokens", location=location)
        temperature = _metadata_float(
            metadata, "mcmc_temperature", location=location
        )
        if (
            steps < 1
            or not steps.is_integer()
            or blocks < 1
            or not blocks.is_integer()
            or not max_tokens.is_integer()
            or int(max_tokens) % int(blocks) != 0
            or temperature != 0.25
        ):
            raise SelectionError(
                f"{location}: invalid corrected MCMC steps/blocks/temperature; "
                "blocks must divide max_tokens"
            )
        expected_config = (
            f"steps{int(steps)}_blocks{int(blocks)}_temp0.25"
        )
    elif candidate.metric_method == "bestofn":
        n = _metadata_float(metadata, "bestofn_n", location=location)
        temperature = _metadata_float(
            metadata, "bestofn_temperature", location=location
        )
        chunk = _metadata_float(metadata, "bestofn_chunk_size", location=location)
        if (
            n < 1
            or not n.is_integer()
            or chunk < 1
            or not chunk.is_integer()
            or temperature != 0.25
        ):
            raise SelectionError(
                f"{location}: invalid corrected Best-of-N n/temperature/chunk"
            )
        expected_config = (
            f"n{int(n)}_temp0.25_chunk{int(chunk)}_lengthnorm"
        )
    else:
        n = _metadata_float(metadata, "majority_n", location=location)
        temperature = _metadata_float(
            metadata, "majority_temperature", location=location
        )
        if n < 1 or not n.is_integer() or temperature != 0.25:
            raise SelectionError(
                f"{location}: invalid corrected majority n/temperature"
            )
        expected_config = f"n{int(n)}_temp0.25"
    if candidate.config != expected_config:
        raise SelectionError(
            f"{location}: {candidate.metric_method} config must equal the "
            f"runner-recorded protocol identity {expected_config!r}"
        )



def _validate_candidate(
    summary: Mapping[str, object], *, index: int, source: str | None = None
) -> Candidate:
    location = _location(index, source)
    missing = [field for field in REQUIRED_FIELDS if field not in summary]
    if missing:
        raise SelectionError(
            f"{location}: missing required fields: {', '.join(missing)}"
        )

    if summary["schema_version"] != 1:
        raise SelectionError(
            f"{location}: unsupported audit schema_version "
            f"{summary['schema_version']!r}; expected 1"
        )

    metric_method = _require_label(summary, "method", location=location).lower()
    config = _require_label(summary, "config", location=location)
    run_mode = _require_label(summary, "run_mode", location=location).lower()
    method = _selection_method(metric_method, run_mode, location=location)
    accuracy = _require_finite_number(summary, "accuracy", location=location)
    mean_c_int = _require_finite_number(summary, "mean_c_int", location=location)
    time_p50_s = _require_finite_number(summary, "time_p50_s", location=location)
    time_p95_s = _require_finite_number(summary, "time_p95_s", location=location)
    n_problems = summary["n_problems"]

    if isinstance(n_problems, bool) or not isinstance(n_problems, int):
        raise SelectionError(
            f"{location}: n_problems must be a positive integer, got {n_problems!r}"
        )
    if n_problems <= 0:
        raise SelectionError(
            f"{location}: n_problems must be a positive integer, got {n_problems!r}"
        )
    if not 0.0 <= accuracy <= 1.0:
        raise SelectionError(
            f"{location}: accuracy must be between 0 and 1, got {accuracy!r}"
        )
    if mean_c_int <= 0:
        raise SelectionError(
            f"{location}: mean_c_int must be positive, got {mean_c_int!r}"
        )
    for field, value in (
        ("time_p50_s", time_p50_s),
        ("time_p95_s", time_p95_s),
    ):
        if value < 0:
            raise SelectionError(
                f"{location}: {field} must be non-negative, got {value!r}"
            )
    if time_p95_s < time_p50_s:
        raise SelectionError(
            f"{location}: time_p95_s must be at least time_p50_s"
        )

    return Candidate(
        metric_method=metric_method,
        method=method,
        run_mode=run_mode,
        config=config,
        accuracy=accuracy,
        mean_c_int=mean_c_int,
        time_p50_s=time_p50_s,
        time_p95_s=time_p95_s,
        n_problems=n_problems,
    )


def load_summary_jsons(paths: Sequence[str | Path]) -> list[dict[str, object]]:
    """Load one audited summary object from each JSON path."""

    if not paths:
        raise SelectionError("at least one summary JSON is required")

    summaries: list[dict[str, object]] = []
    for path_like in paths:
        path = Path(path_like)
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except OSError as exc:
            raise SelectionError(f"cannot read summary JSON {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SelectionError(f"invalid summary JSON {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise SelectionError(f"{path}: summary JSON root must be an object")
        if _ARTIFACT_KEY in value:
            raise SelectionError(
                f"{path}: reserved field {_ARTIFACT_KEY!r} may not appear in an audit summary"
            )
        value[_ARTIFACT_KEY] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "audit_schema_version": value.get("schema_version"),
        }
        summaries.append(value)
    return summaries


def _validate_budgets(budgets: Sequence[float]) -> list[float]:
    if not budgets:
        raise SelectionError("at least one absolute C_int budget is required")
    validated: list[float] = []
    seen: set[float] = set()
    for index, value in enumerate(budgets):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SelectionError(f"budget[{index}] must be numeric, got {value!r}")
        budget = float(value)
        if not math.isfinite(budget) or budget <= 0:
            raise SelectionError(
                f"budget[{index}] must be finite and positive, got {value!r}"
            )
        if budget in seen:
            raise SelectionError(f"duplicate budget: {budget:g}")
        seen.add(budget)
        validated.append(budget)
    return sorted(validated)


def _validated_source_artifacts(
    summary: Mapping[str, object], *, location: str
) -> list[dict[str, str]]:
    artifacts = summary.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SelectionError(
            f"{location}: publication audit summary must identify raw CSV "
            "source_artifacts"
        )
    validated: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        artifact_location = f"{location}.source_artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            raise SelectionError(f"{artifact_location} must be an object")
        path_value = artifact.get("path")
        sha_value = artifact.get("sha256")
        if not isinstance(path_value, str) or not path_value.strip():
            raise SelectionError(f"{artifact_location}.path is invalid")
        path_text = path_value.strip()
        if path_text in seen_paths:
            raise SelectionError(f"{location}: duplicate raw CSV path {path_text!r}")
        seen_paths.add(path_text)
        if (
            not isinstance(sha_value, str)
            or len(sha_value) != 64
            or any(char not in "0123456789abcdef" for char in sha_value.lower())
        ):
            raise SelectionError(
                f"{artifact_location}.sha256 must be 64 hexadecimal digits"
            )
        source_path = Path(path_text)
        if not source_path.is_file():
            raise SelectionError(
                f"{artifact_location} raw CSV does not exist: {source_path}"
            )
        observed_sha = _sha256_file(source_path)
        if observed_sha.lower() != sha_value.lower():
            raise SelectionError(
                f"{artifact_location} raw CSV SHA-256 does not match: {source_path}"
            )
        validated.append({"path": path_text, "sha256": sha_value.lower()})

    source_files = summary.get("source_files")
    if source_files != [artifact["path"] for artifact in validated]:
        raise SelectionError(
            f"{location}: source_files must exactly match source_artifacts paths"
        )
    if summary.get("n_files") != len(validated):
        raise SelectionError(
            f"{location}: n_files does not match source_artifacts"
        )
    return validated


def _verify_audit_derivation(
    summary: Mapping[str, object],
    candidate: Candidate,
    source_artifacts: Sequence[Mapping[str, str]],
    *,
    location: str,
) -> None:
    """Re-run raw CSV -> audit summary instead of trusting self-certified JSON."""

    try:
        from .result_audit import AuditError, audit_csvs
    except ImportError:  # Direct execution from the repository root.
        from result_audit import AuditError, audit_csvs  # type: ignore

    budget_cap = summary.get("budget_cap")
    budget_tolerance = summary.get("budget_tolerance")
    if budget_tolerance is None:
        budget_tolerance = 1.02
    try:
        recomputed = audit_csvs(
            [artifact["path"] for artifact in source_artifacts],
            method=candidate.metric_method,
            config=candidate.config,
            run_mode=candidate.run_mode,
            expected_problem_count=candidate.n_problems,
            allow_legacy_aliases=False,
            budget_cap=budget_cap,
            budget_tolerance=budget_tolerance,
            require_provenance=True,
        )
    except (AuditError, OSError, TypeError, ValueError) as exc:
        raise SelectionError(
            f"{location}: raw CSV -> audit summary derivation failed: {exc}"
        ) from exc

    recorded = dict(summary)
    recorded.pop(_ARTIFACT_KEY, None)
    if recomputed != recorded:
        raise SelectionError(
            f"{location}: audit summary does not equal the result re-derived "
            "from its hash-checked raw CSV files"
        )


def select_compute_matched(
    summaries: Sequence[Mapping[str, object]],
    budgets: Sequence[float],
    *,
    tolerance: float = 1.02,
    require_provenance: bool = False,
    expected_problem_count: int = 500,
) -> dict[str, object]:
    """Validate candidates and select one configuration per series and budget.

    A series with no configuration under a cap is retained in the output with
    ``status="no_eligible_config"`` and null metrics.  This makes incomplete
    compute-matched comparisons visible in both JSON and CSV output.
    """

    if not summaries:
        raise SelectionError("at least one result summary is required")
    if (
        isinstance(expected_problem_count, bool)
        or not isinstance(expected_problem_count, int)
        or expected_problem_count <= 0
    ):
        raise SelectionError("expected_problem_count must be a positive integer")
    if require_provenance and expected_problem_count != 500:
        raise SelectionError(
            "corrected-paper-v1 publication selection requires exactly 500 problems"
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise SelectionError(f"tolerance must be numeric, got {tolerance!r}")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 1.0:
        raise SelectionError(
            f"tolerance must be finite and at least 1.0, got {tolerance!r}"
        )
    if require_provenance and not math.isclose(
        tolerance, 1.02, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SelectionError(
            "corrected-paper-v1 requires budget_tolerance=1.02"
        )
    validated_budgets = _validate_budgets(budgets)

    candidates: list[Candidate] = []
    pairs: set[tuple[str, str]] = set()
    expected_n_problems: int | None = None
    metadata_values: dict[str, set[str]] = {
        field: set() for field in COMPARABILITY_FIELDS
    }
    metadata_coverage: dict[str, int] = {
        field: 0 for field in COMPARABILITY_FIELDS
    }
    input_summaries: list[dict[str, object]] = []
    adaptive_instance_caps: dict[Candidate, float] = {}
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping):
            raise SelectionError(f"summary[{index}] must be an object")
        candidate = _validate_candidate(summary, index=index)
        pair = (candidate.method, candidate.config)
        if pair in pairs:
            raise SelectionError(
                "duplicate method/config pair: "
                f"method={candidate.method!r}, config={candidate.config!r}"
            )
        pairs.add(pair)
        if expected_n_problems is None:
            expected_n_problems = candidate.n_problems
        elif candidate.n_problems != expected_n_problems:
            raise SelectionError(
                "n_problems mismatch: "
                f"expected {expected_n_problems}, got {candidate.n_problems} "
                f"for method={candidate.method!r}, config={candidate.config!r}"
            )
        if candidate.n_problems != expected_problem_count:
            raise SelectionError(
                f"summary[{index}]: expected {expected_problem_count} benchmark "
                f"problems, got {candidate.n_problems} for "
                f"method={candidate.method!r}, config={candidate.config!r}"
            )
        candidates.append(candidate)

        location = f"summary[{index}]"
        if require_provenance and summary.get("provenance_complete") is not True:
            raise SelectionError(
                f"{location}: publication provenance is not complete"
            )
        if (
            require_provenance
            and summary.get("publication_provenance_required") is not True
        ):
            raise SelectionError(
                f"{location}: audit summary was not produced with "
                "--require-provenance"
            )

        invariant_metadata = summary.get("invariant_metadata", {})
        if invariant_metadata is None:
            invariant_metadata = {}
        if not isinstance(invariant_metadata, Mapping):
            raise SelectionError(
                f"{location}: invariant_metadata must be an object"
            )
        if require_provenance:
            _validate_common_publication_profile(
                invariant_metadata, location=location
            )
            _validate_asmc_publication_profile(
                summary,
                candidate,
                invariant_metadata,
                location=location,
            )
            _validate_baseline_publication_profile(
                candidate,
                invariant_metadata,
                location=location,
            )
            if candidate.method == "asmc-adaptive":
                adaptive_instance_caps[candidate] = float(
                    invariant_metadata["asmc_c_int_cap"]
                )

        artifact = summary.get(_ARTIFACT_KEY)
        if artifact is not None:
            if not isinstance(artifact, Mapping):
                raise SelectionError(
                    f"{location}: {_ARTIFACT_KEY} must be an object"
                )
            artifact_path = artifact.get("path")
            artifact_sha = artifact.get("sha256")
            artifact_schema = artifact.get("audit_schema_version")
            if not isinstance(artifact_path, str) or not artifact_path.strip():
                raise SelectionError(
                    f"{location}: audit summary artifact path is invalid"
                )
            if (
                not isinstance(artifact_sha, str)
                or len(artifact_sha) != 64
                or any(char not in "0123456789abcdef" for char in artifact_sha.lower())
            ):
                raise SelectionError(
                    f"{location}: audit summary artifact SHA-256 is invalid"
                )
            if artifact_schema != 1:
                raise SelectionError(
                    f"{location}: audit summary artifact schema must be 1"
                )
            source_artifacts: list[dict[str, str]] = []
            if require_provenance:
                source_artifacts = _validated_source_artifacts(
                    summary, location=location
                )
                _verify_audit_derivation(
                    summary,
                    candidate,
                    source_artifacts,
                    location=location,
                )
            input_summaries.append(
                {
                    "path": artifact_path.strip(),
                    "sha256": artifact_sha.lower(),
                    "audit_schema_version": artifact_schema,
                    "method": candidate.method,
                    "metric_method": candidate.metric_method,
                    "run_mode": candidate.run_mode,
                    "config": candidate.config,
                    "source_artifacts": source_artifacts,
                }
            )
        elif require_provenance:
            raise SelectionError(
                f"{location}: publication selection requires the path and "
                "SHA-256 of each audit summary; load summaries from JSON files"
            )

        for field in COMPARABILITY_FIELDS:
            value = invariant_metadata.get(field)
            if value is None or not str(value).strip():
                continue
            metadata_values[field].add(str(value).strip())
            metadata_coverage[field] += 1

    conflicting_metadata = {
        field: sorted(values)
        for field, values in metadata_values.items()
        if len(values) > 1
    }
    if conflicting_metadata:
        details = "; ".join(
            f"{field}={values!r}"
            for field, values in sorted(conflicting_metadata.items())
        )
        raise SelectionError(
            "candidate summaries use incompatible run metadata: " + details
        )

    incomplete_comparability_fields = sorted(
        field
        for field, coverage in metadata_coverage.items()
        if coverage != len(summaries)
    )
    if require_provenance and incomplete_comparability_fields:
        raise SelectionError(
            "candidate summaries lack comparable metadata for: "
            + ", ".join(incomplete_comparability_fields)
        )
    common_metadata = {
        field: next(iter(values))
        for field, values in metadata_values.items()
        if len(values) == 1 and metadata_coverage[field] == len(summaries)
    }

    by_method: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_method.setdefault(candidate.method, []).append(candidate)

    required_series = [
        "asmc",
        "asmc-adaptive",
        "bestofn",
        "greedy",
        "mcmc",
        "naive",
    ]
    if require_provenance:
        missing_series = sorted(set(required_series) - set(by_method))
        unexpected_series = sorted(set(by_method) - set(required_series))
        if missing_series or unexpected_series:
            details = []
            if missing_series:
                details.append("missing: " + ", ".join(missing_series))
            if unexpected_series:
                details.append("unexpected: " + ", ".join(unexpected_series))
            raise SelectionError(
                "corrected-paper-v1 publication roster mismatch ("
                + "; ".join(details)
                + ")"
            )

    budget_baseline: dict[str, object] | None = None
    budget_multipliers: list[int] | None = None
    if require_provenance:
        naive_candidates = by_method.get("naive", [])
        if len(naive_candidates) != 1:
            raise SelectionError(
                "corrected-paper-v1 requires exactly one audited naive "
                "single-sample baseline to derive C0"
            )
        naive_baseline = naive_candidates[0]
        c0 = naive_baseline.mean_c_int
        allowed_multipliers = (2, 4, 8, 16, 32, 64, 128)
        budget_multipliers = []
        for budget in validated_budgets:
            matching = [
                multiplier
                for multiplier in allowed_multipliers
                if math.isclose(
                    budget,
                    multiplier * c0,
                    rel_tol=1e-12,
                    abs_tol=1e-6,
                )
            ]
            if len(matching) != 1:
                raise SelectionError(
                    f"publication budget {budget:g} is not one of "
                    "{2,4,8,16,32,64,128} times the audited naive C0 "
                    f"({c0:g})"
                )
            budget_multipliers.append(matching[0])
        budget_baseline = {
            "method": "naive",
            "config": naive_baseline.config,
            "mean_c_int": c0,
        }

    budget_reports: list[dict[str, object]] = []
    for budget in validated_budgets:
        cap = budget * tolerance
        selections: list[dict[str, object]] = []
        for method in sorted(by_method):
            series_candidates = by_method[method]
            metric_method = series_candidates[0].metric_method
            run_mode = series_candidates[0].run_mode
            eligible = [
                candidate
                for candidate in series_candidates
                if candidate.mean_c_int <= cap
                and (
                    not require_provenance
                    or method != "asmc-adaptive"
                    or math.isclose(
                        adaptive_instance_caps[candidate],
                        budget,
                        rel_tol=1e-12,
                        abs_tol=1e-6,
                    )
                )
            ]
            if not eligible:
                selections.append(
                    {
                        "method": method,
                        "metric_method": metric_method,
                        "run_mode": run_mode,
                        "status": "no_eligible_config",
                        "eligible_config_count": 0,
                        "config": None,
                        "accuracy": None,
                        "mean_c_int": None,
                        "time_p50_s": None,
                        "time_p95_s": None,
                        "per_instance_c_int_cap": None,
                        "n_problems": expected_n_problems,
                    }
                )
                continue

            selected = min(
                eligible,
                key=lambda candidate: (
                    -candidate.accuracy,
                    candidate.mean_c_int,
                    candidate.time_p95_s,
                    candidate.config,
                ),
            )
            selected_row = selected.as_dict()
            selected_row.update(
                {
                    "status": "selected",
                    "eligible_config_count": len(eligible),
                    "per_instance_c_int_cap": (
                        adaptive_instance_caps[selected]
                        if require_provenance and method == "asmc-adaptive"
                        else None
                    ),
                }
            )
            selections.append(selected_row)

        budget_reports.append(
            {
                "budget_c_int": budget,
                "cap_c_int": cap,
                "selections": selections,
            }
        )

    return {
        "schema_version": 1,
        "budget_tolerance": tolerance,
        "n_problems": expected_n_problems,
        "expected_problem_count": expected_problem_count,
        "budget_baseline": budget_baseline,
        "budget_multipliers": budget_multipliers,
        "n_candidates": len(candidates),
        "input_summaries": input_summaries,
        "methods": sorted(by_method),
        "required_publication_series": (
            required_series if require_provenance else []
        ),
        "publication_provenance_required": require_provenance,
        "raw_audit_derivation_verified": require_provenance,
        "asmc_publication_profile": (
            ASMC_PUBLICATION_PROFILE if require_provenance else "not-enforced"
        ),
        "comparability_metadata_complete": not incomplete_comparability_fields,
        "incomplete_comparability_fields": incomplete_comparability_fields,
        "common_invariant_metadata": dict(sorted(common_metadata.items())),
        "budgets": budget_reports,
    }


def write_json_selection(report: Mapping[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(output_path)


def write_csv_selection(report: Mapping[str, object], path: str | Path) -> None:
    """Write one flat row per budget/method selection."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "schema_version",
        "budget_c_int",
        "budget_tolerance",
        "cap_c_int",
        "method",
        "metric_method",
        "run_mode",
        "status",
        "eligible_config_count",
        "config",
        "accuracy",
        "mean_c_int",
        "time_p50_s",
        "time_p95_s",
        "per_instance_c_int_cap",
        "n_problems",
    ]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for budget_report in report["budgets"]:
            for selection in budget_report["selections"]:
                writer.writerow(
                    {
                        "schema_version": report["schema_version"],
                        "budget_c_int": budget_report["budget_c_int"],
                        "budget_tolerance": report["budget_tolerance"],
                        "cap_c_int": budget_report["cap_c_int"],
                        **selection,
                    }
                )
    temporary_path.replace(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best audited configuration per publication series "
            "under one or more absolute C_int budgets."
        )
    )
    parser.add_argument(
        "summary_json",
        nargs="+",
        type=Path,
        help="configuration-level JSON summaries from result_audit.py",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help=(
            "re-derive every audit from hash-checked raw CSVs, require the "
            "corrected paper profile, and share complete run metadata"
        ),
    )
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument(
        "--budget",
        action="append",
        nargs="+",
        type=float,
        metavar="C_INT",
        help=(
            "absolute C_int budget; accept multiple values after one flag or "
            "repeat --budget"
        ),
    )
    budget_group.add_argument(
        "--budget-multiplier",
        action="append",
        nargs="+",
        type=int,
        metavar="MULTIPLE",
        help=(
            "derive absolute budgets from the audited naive C0; publication "
            "multipliers are 2, 4, 8, 16, 32, 64, and 128"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.02,
        help="multiplicative budget tolerance (default: 1.02)",
    )
    parser.add_argument(
        "--expected-problems",
        type=int,
        default=500,
        help=(
            "required problem count for every audited summary (default: 500); "
            "set explicitly only for diagnostic subsets"
        ),
    )
    parser.add_argument("--json-out", type=Path, help="write the selection as JSON")
    parser.add_argument("--csv-out", type=Path, help="write flat selection rows as CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        input_paths = {path.resolve() for path in args.summary_json}
        output_paths = [
            path.resolve()
            for path in (args.json_out, args.csv_out)
            if path is not None
        ]
        output_write_paths = []
        for path in output_paths:
            output_write_paths.extend(
                [path, path.with_suffix(path.suffix + ".tmp")]
            )
        if len(set(output_write_paths)) != len(output_write_paths):
            raise SelectionError(
                "selection output paths and their atomic temporary paths must "
                "all be distinct"
            )
        summaries = load_summary_jsons(args.summary_json)
        repository_root = Path(__file__).resolve().parents[1]
        dependency_paths = {
            *input_paths,
            Path(__file__).resolve(),
            (repository_root / "analysis" / "result_audit.py").resolve(),
            (repository_root / "data" / "MATH500.json").resolve(),
        }
        for summary in summaries:
            artifacts = summary.get("source_artifacts", [])
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if isinstance(artifact, Mapping):
                        path_value = artifact.get("path")
                        if isinstance(path_value, str) and path_value.strip():
                            dependency_paths.add(Path(path_value).resolve())
            benchmark = summary.get("benchmark_artifact")
            if isinstance(benchmark, Mapping):
                path_value = benchmark.get("path")
                if isinstance(path_value, str) and path_value.strip():
                    benchmark_path = Path(path_value)
                    if not benchmark_path.is_absolute():
                        benchmark_path = repository_root / benchmark_path
                    dependency_paths.add(benchmark_path.resolve())
        overlapping_paths = dependency_paths.intersection(output_write_paths)
        if overlapping_paths:
            raise SelectionError(
                "an output path must not overwrite an audit summary, raw CSV, "
                "or pinned benchmark dependency: "
                + ", ".join(str(path) for path in sorted(overlapping_paths))
            )
        dependency_hashes = (
            {
                path: _sha256_file(path)
                for path in dependency_paths
                if path.is_file()
            }
            if args.require_provenance
            else {}
        )
        if args.budget_multiplier is not None:
            multipliers = [
                value for group in args.budget_multiplier for value in group
            ]
            naive = [
                summary
                for summary in summaries
                if str(summary.get("method", "")).strip().lower() == "naive"
            ]
            if len(naive) != 1:
                raise SelectionError(
                    "--budget-multiplier requires exactly one audited naive C0 summary"
                )
            c0 = _require_finite_number(
                naive[0], "mean_c_int", location="naive C0 summary"
            )
            budgets = [c0 * multiplier for multiplier in multipliers]
        else:
            budgets = [value for group in args.budget for value in group]
        report = select_compute_matched(
            summaries,
            budgets,
            tolerance=args.tolerance,
            require_provenance=args.require_provenance,
            expected_problem_count=args.expected_problems,
        )
        if args.json_out is not None:
            write_json_selection(report, args.json_out)
        if args.csv_out is not None:
            write_csv_selection(report, args.csv_out)
        for path in output_paths:
            if not path.is_file():
                raise SelectionError(f"selection output was not created: {path}")
        for path, expected_sha256 in dependency_hashes.items():
            if not path.is_file() or _sha256_file(path) != expected_sha256:
                raise SelectionError(
                    f"publication input changed while writing outputs: {path}"
                )
    except SelectionError as exc:
        parser.error(str(exc))

    if args.json_out is None and args.csv_out is None:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
