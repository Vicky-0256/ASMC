#!/usr/bin/env python3
"""Validate and summarize per-problem result CSV files.

The paper evaluation uses exactly one row for every MATH500 problem.  This
module intentionally treats violations of that invariant as errors instead of
silently averaging partial, duplicated, or mixed-protocol runs.

Legacy experiment CSVs do not contain trustworthy code-version metadata.
Consequently, this tool never guesses or injects a Git commit SHA.  Method and
configuration labels are supplied explicitly and are checked against matching
CSV metadata columns when those columns exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence


class AuditError(ValueError):
    """Raised when result artifacts fail a reproducibility invariant."""


PUBLICATION_METADATA_COLUMNS = (
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

_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PUBLICATION_COMPUTE_SCHEMA = "asmc-compute-v2"
PUBLICATION_TIMING_SCHEMA = "synchronized-end-to-end-wall-clock-v1"
PUBLICATION_RNG_PROTOCOL = "sha256-canonical-json-u32-method-isolation-v1"
MATH500_DATASET_NAME = "MATH500"
MATH500_DATASET_SHA256 = (
    "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
)
MATH500_BATCH_SIZE = 100
MATH500_PATH = Path(__file__).resolve().parents[1] / "data" / "MATH500.json"
PUBLICATION_MODEL_ID = "Qwen/Qwen2.5-Math-7B"
QWEN25_MATH_7B_EOS_TOKEN_IDS = [151643]
# Pinned from Qwen/Qwen2.5-Math-7B's model output ``vocab_size``.  This is the
# logits dimension, not a claim about how many entries a tokenizer exposes.
QWEN25_MATH_7B_OUTPUT_VOCAB_SIZE = 152064
ASMC_HARD_EPSILON = 0.08
GENERATION_METHODS = frozenset(
    {"greedy", "naive", "std", "majority", "mcmc", "bestofn"}
)
PUBLICATION_METHOD_PROTOCOLS = {
    "greedy": "deterministic-greedy-decoding-v2",
    "asmc": "cache-coherent-asmc-corrected-v1",
    "naive": "single-temperature-sample-v2",
    "std": "single-temperature-one-sample-v2",
    "mcmc": "completion-only-eos-mcmc-power-sampling-v4",
    "majority": "independent-sampling-unweighted-answer-majority-v2",
    "bestofn": (
        "independent-generation-unconditional-length-normalized-"
        "logprob-argmax-v3"
    ),
}
PUBLICATION_SAMPLING_POLICY = "full-support-temperature-only-v1"
# Keep this stdlib-only mirror explicit: ``result_audit.py`` must remain usable
# with ``python -S`` on a release machine that has no torch/transformers.  A
# regression test compares it with the runtime's exported contract so the two
# definitions cannot drift silently.
PUBLICATION_FULL_SUPPORT_GENERATION_KWARGS: dict[str, object] = {
    "top_k": 0,
    "top_p": 1.0,
    "typical_p": 1.0,
    "min_p": None,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "use_cache": True,
    "guidance_scale": 1.0,
    "sequence_bias": None,
    "diversity_penalty": 0.0,
    "encoder_repetition_penalty": 1.0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "min_length": 0,
    "min_new_tokens": 0,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "forced_decoder_ids": None,
    "remove_invalid_values": False,
    "exponential_decay_length_penalty": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "watermarking_config": None,
    "renormalize_logits": False,
    "num_beams": 1,
    "num_beam_groups": 1,
    "penalty_alpha": None,
    "constraints": None,
    "force_words_ids": None,
    "num_return_sequences": 1,
    "max_time": None,
    "stop_strings": None,
    "token_healing": False,
    "dola_layers": None,
}
PUBLICATION_RESOLVED_GENERATION_KWARGS = {
    **PUBLICATION_FULL_SUPPORT_GENERATION_KWARGS,
    "eos_token_id": QWEN25_MATH_7B_EOS_TOKEN_IDS,
    "pad_token_id": QWEN25_MATH_7B_EOS_TOKEN_IDS[0],
}
PUBLICATION_SAMPLING_POLICY_PAYLOAD = json.dumps(
    PUBLICATION_RESOLVED_GENERATION_KWARGS,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
)
PUBLICATION_SAMPLING_PROTOCOL_METADATA: dict[str, object] = {
    "sampling_policy": PUBLICATION_SAMPLING_POLICY,
    **{
        f"sampling_{name}": "none" if value is None else value
        for name, value in PUBLICATION_FULL_SUPPORT_GENERATION_KWARGS.items()
    },
    "sampling_eos_token_ids": json.dumps(
        QWEN25_MATH_7B_EOS_TOKEN_IDS, separators=(",", ":")
    ),
    "sampling_pad_token_id": QWEN25_MATH_7B_EOS_TOKEN_IDS[0],
    "sampling_policy_payload": PUBLICATION_SAMPLING_POLICY_PAYLOAD,
    "sampling_policy_sha256": hashlib.sha256(
        PUBLICATION_SAMPLING_POLICY_PAYLOAD.encode("utf-8")
    ).hexdigest(),
}
PUBLICATION_SAMPLING_METADATA_COLUMNS = tuple(
    PUBLICATION_SAMPLING_PROTOCOL_METADATA
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_math500() -> tuple[list[dict[str, str]], dict[str, object]]:
    """Load the repository benchmark only after checking its pinned digest."""

    try:
        benchmark_bytes = MATH500_PATH.read_bytes()
    except OSError as exc:
        raise AuditError(
            f"cannot read pinned MATH500 benchmark: {MATH500_PATH}"
        ) from exc
    observed_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    if observed_sha256.lower() != MATH500_DATASET_SHA256:
        raise AuditError(
            "repository MATH500.json does not match the publication pin: "
            f"expected {MATH500_DATASET_SHA256}, got {observed_sha256}"
        )

    try:
        raw_rows = json.loads(benchmark_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(
            f"cannot parse pinned MATH500 benchmark: {MATH500_PATH}"
        ) from exc
    if not isinstance(raw_rows, list) or len(raw_rows) != 500:
        raise AuditError("pinned MATH500 benchmark must contain exactly 500 rows")

    rows: list[dict[str, str]] = []
    seen_dataset_ids: set[str] = set()
    for problem_idx, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise AuditError(
                f"pinned MATH500 row {problem_idx} must be a JSON object"
            )
        canonical_row: dict[str, str] = {}
        for field in ("id", "prompt", "answer"):
            value = raw_row.get(field)
            if not isinstance(value, str) or not value:
                raise AuditError(
                    f"pinned MATH500 row {problem_idx} has invalid {field!r}"
                )
            canonical_row[field] = value
        if canonical_row["id"] in seen_dataset_ids:
            raise AuditError(
                f"pinned MATH500 has duplicate dataset id {canonical_row['id']!r}"
            )
        seen_dataset_ids.add(canonical_row["id"])
        rows.append(canonical_row)

    return rows, {
        "path": "data/MATH500.json",
        "sha256": observed_sha256,
        "n_rows": len(rows),
    }


def _load_math_grading_functions():
    """Import the runner's parser/grader lazily for strict audits only."""

    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    try:
        from grader_utils.math_grader import grade_answer
        from grader_utils.parse_utils import parse_answer, parse_answer_robust
    except (ImportError, ModuleNotFoundError) as exc:
        raise AuditError(
            "strict publication audit requires the repository math grader "
            "and answer parser"
        ) from exc
    return grade_answer, parse_answer, parse_answer_robust


def _grade_with_runner_fallback(
    given_answer: str | None,
    ground_truth: str,
    *,
    grade_answer,
) -> bool:
    """Mirror the runner's grade-then-exact-string fallback policy."""

    if given_answer is None:
        return False
    try:
        return bool(grade_answer(given_answer, ground_truth))
    except Exception:
        return str(given_answer).strip() == str(ground_truth).strip()


def _parse_completion_token_ids(
    row: Mapping[str, str], *, method: str, location: str
) -> tuple[list[int], bool]:
    """Validate the raw completion-token evidence for a strict audit row."""

    column = f"{method}_completion_token_ids"
    serialized = _required_row_text(row, column=column, location=location)
    try:
        token_ids = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{location}: {column} is not valid JSON") from exc
    if not isinstance(token_ids, list):
        raise AuditError(f"{location}: {column} must be a JSON list")
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise AuditError(
                f"{location}: {column}[{index}] must be an integer token ID"
            )
        if token_id < 0:
            raise AuditError(
                f"{location}: {column}[{index}] must be non-negative"
            )
        if token_id >= QWEN25_MATH_7B_OUTPUT_VOCAB_SIZE:
            raise AuditError(
                f"{location}: {column}[{index}]={token_id} is outside the "
                "pinned Qwen2.5-Math-7B output vocabulary range "
                f"0..{QWEN25_MATH_7B_OUTPUT_VOCAB_SIZE - 1}"
            )
    canonical = json.dumps(token_ids, separators=(",", ":"))
    if serialized != canonical:
        raise AuditError(
            f"{location}: {column} must use canonical compact JSON; "
            f"expected {canonical!r}"
        )

    eos_token_id = QWEN25_MATH_7B_EOS_TOKEN_IDS[0]
    actual_has_eos = eos_token_id in token_ids
    if actual_has_eos and token_ids.index(eos_token_id) != len(token_ids) - 1:
        raise AuditError(
            f"{location}: {column} must contain no token after its first EOS"
        )

    has_eos_column = f"{method}_completion_has_eos"
    reported_has_eos = _parse_bool(
        row.get(has_eos_column, ""),
        column=has_eos_column,
        location=location,
    )
    if reported_has_eos != actual_has_eos:
        raise AuditError(
            f"{location}: {has_eos_column}={reported_has_eos} does not match "
            f"the token artifact ({actual_has_eos})"
        )

    max_tokens = _parse_integral_csv_number(
        row.get("max_tokens", ""), column="max_tokens", location=location
    )
    if len(token_ids) > max_tokens:
        raise AuditError(
            f"{location}: {column} exceeds max_tokens={max_tokens}"
        )
    if not token_ids and method != "asmc":
        raise AuditError(
            f"{location}: an empty token artifact is allowed only for ASMC "
            "pre-generation budget exhaustion"
        )
    if not actual_has_eos:
        if method != "asmc":
            if len(token_ids) != max_tokens:
                raise AuditError(
                    f"{location}: a non-EOS {method} completion must contain "
                    f"exactly max_tokens={max_tokens} token IDs"
                )
        else:
            stop_reason = str(row.get("asmc_stop_reason", "")).strip()
            if not token_ids:
                budget_exhausted = _parse_bool(
                    row.get("asmc_budget_exhausted", ""),
                    column="asmc_budget_exhausted",
                    location=location,
                )
                exhausted_at = _parse_integral_csv_number(
                    row.get("asmc_budget_exhausted_at_token", ""),
                    column="asmc_budget_exhausted_at_token",
                    location=location,
                )
                if (
                    not budget_exhausted
                    or stop_reason != "budget_exhausted"
                    or exhausted_at != -1
                ):
                    raise AuditError(
                        f"{location}: an empty ASMC token artifact is allowed "
                        "only for pre-generation budget exhaustion at token -1"
                    )
            elif stop_reason == "max_len" and len(token_ids) != max_tokens:
                raise AuditError(
                    f"{location}: ASMC stop_reason='max_len' requires exactly "
                    f"max_tokens={max_tokens} token IDs when EOS is absent"
                )
            elif stop_reason not in {
                "early_stop",
                "all_finished",
                "budget_exhausted",
                "max_len",
            }:
                raise AuditError(
                    f"{location}: non-EOS ASMC completion has invalid "
                    f"stop_reason {stop_reason!r}"
                )
    return token_ids, actual_has_eos


def _verify_publication_ground_truth_row(
    row: Mapping[str, str],
    *,
    method: str,
    problem_idx: int,
    canonical_row: Mapping[str, str],
    location: str,
) -> tuple[bool, str]:
    """Verify dataset identity and independently recompute row correctness."""

    question = str(row.get("question", ""))
    if question != canonical_row["prompt"]:
        raise AuditError(
            f"{location}: question does not match pinned MATH500 problem_idx "
            f"{problem_idx}"
        )
    ground_truth = str(row.get("correct_answer", ""))
    if ground_truth != canonical_row["answer"]:
        raise AuditError(
            f"{location}: correct_answer does not match pinned MATH500 "
            f"problem_idx {problem_idx}"
        )

    batch_idx = _parse_integral_csv_number(
        row.get("batch_idx", ""), column="batch_idx", location=location
    )
    expected_batch_idx = problem_idx // MATH500_BATCH_SIZE
    if batch_idx != expected_batch_idx:
        raise AuditError(
            f"{location}: batch_idx={batch_idx} is inconsistent with "
            f"problem_idx={problem_idx}; expected {expected_batch_idx}"
        )

    for id_column in ("dataset_problem_id", "dataset_id"):
        dataset_id = row.get(id_column)
        if dataset_id is not None and str(dataset_id).strip():
            if str(dataset_id) != canonical_row["id"]:
                raise AuditError(
                    f"{location}: {id_column} does not match pinned MATH500 "
                    f"problem_idx {problem_idx}"
                )

    answer_column = f"{method}_answer"
    completion_column = f"{method}_completion"
    token_ids, _ = _parse_completion_token_ids(
        row, method=method, location=location
    )
    stored_answer = row.get(answer_column)
    if completion_column not in row or row.get(completion_column) is None:
        raise AuditError(
            f"{location}: {completion_column} must be present; an empty "
            "decoded completion must be stored as an empty string"
        )
    completion = row.get(completion_column)
    stored_answer_text = "" if stored_answer is None else str(stored_answer).strip()
    completion_text = "" if completion is None else str(completion).strip()
    pre_generation_budget_exhaustion = method == "asmc" and not token_ids
    if not completion_text:
        if stored_answer_text:
            raise AuditError(
                f"{location}: empty {completion_column} cannot have a "
                f"populated {answer_column}"
            )
        # Token IDs can decode to an empty/whitespace-only string for several
        # legitimate reasons (special tokens, whitespace plus EOS, or a
        # max-length sequence).  This stdlib-only audit intentionally does not
        # claim to reproduce tokenizer decoding.  Structural validation above
        # is the evidence available here; an empty decoded answer is incorrect.
        return False, "completion_token_ids"
    if pre_generation_budget_exhaustion:
        raise AuditError(
            f"{location}: non-empty {completion_column} is inconsistent with "
            "pre-generation ASMC budget exhaustion"
        )
    if token_ids == QWEN25_MATH_7B_EOS_TOKEN_IDS:
        raise AuditError(
            f"{location}: non-empty {completion_column} is inconsistent with "
            "an EOS-only token artifact"
        )
    if completion_text.startswith("ERROR:"):
        raise AuditError(
            f"{location}: {completion_column} is an error artifact and cannot "
            "support publication correctness"
        )

    grade_answer, parse_answer, parse_answer_robust = (
        _load_math_grading_functions()
    )
    # The corrected MATH500 protocol uses the same robust repository parser
    # for every method; method-specific parsing would make correctness depend
    # on the table series rather than the emitted completion.
    parsed_answer = parse_answer_robust(completion_text)
    candidate_answer = None if parsed_answer is None else str(parsed_answer).strip()
    if stored_answer_text:
        if candidate_answer is None:
            raise AuditError(
                f"{location}: {answer_column} is populated but the repository "
                f"parser cannot recover it from {completion_column}"
            )
        answer_matches_completion = (
            str(candidate_answer).strip() == stored_answer_text
            or _grade_with_runner_fallback(
                candidate_answer,
                stored_answer_text,
                grade_answer=grade_answer,
            )
            or _grade_with_runner_fallback(
                stored_answer_text,
                candidate_answer,
                grade_answer=grade_answer,
            )
        )
        if not answer_matches_completion:
            raise AuditError(
                f"{location}: {answer_column} is inconsistent with the "
                f"repository parse of {completion_column}"
            )

    recomputed_correct = _grade_with_runner_fallback(
        candidate_answer,
        canonical_row["answer"],
        grade_answer=grade_answer,
    )
    return recomputed_correct, "completion"


def _required_row_text(
    row: Mapping[str, str], *, column: str, location: str
) -> str:
    value = row.get(column)
    if value is None or not str(value).strip():
        raise AuditError(f"{location}: {column} must be non-empty")
    return str(value).strip()


def _publication_sampling_config(
    row: Mapping[str, str], *, location: str
) -> dict[str, object]:
    """Parse and freeze the full-support stochastic generation contract.

    Optional values use the runner's explicit, non-empty ``none`` sentinel in
    CSV.  The returned typed values intentionally match the runner's canonical
    RNG identity exactly.
    """

    missing = [
        field for field in PUBLICATION_SAMPLING_METADATA_COLUMNS if field not in row
    ]
    if missing:
        raise AuditError(
            f"{location}: missing full-support sampling metadata: "
            + ", ".join(missing)
        )

    use_cache = _parse_bool(
        row["sampling_use_cache"],
        column="sampling_use_cache",
        location=location,
    )
    if not use_cache:
        raise AuditError(
            f"{location}: full-support sampling requires sampling_use_cache=True"
        )
    for field, expected in PUBLICATION_SAMPLING_PROTOCOL_METADATA.items():
        raw_value = row[field]
        if isinstance(expected, bool):
            observed: object = _parse_bool(
                raw_value, column=field, location=location
            )
        elif isinstance(expected, int):
            observed = _parse_integral_csv_number(
                raw_value, column=field, location=location
            )
        elif isinstance(expected, float):
            observed = _parse_nonnegative_float(
                raw_value, column=field, location=location
            )
        else:
            observed = str(raw_value).strip()

        matches = observed == expected
        if not matches:
            raise AuditError(
                f"{location}: full-support sampling requires "
                f"{field}={expected!r}, got {observed!r}"
            )

    payload_text = str(row["sampling_policy_payload"]).strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise AuditError(
            f"{location}: sampling_policy_payload is not valid JSON"
        ) from exc
    if payload != PUBLICATION_RESOLVED_GENERATION_KWARGS:
        raise AuditError(
            f"{location}: sampling_policy_payload does not encode the "
            "canonical resolved full-support generation contract"
        )
    if payload_text != PUBLICATION_SAMPLING_POLICY_PAYLOAD:
        raise AuditError(
            f"{location}: sampling_policy_payload must use canonical compact "
            "JSON with sorted keys"
        )
    observed_sha = str(row["sampling_policy_sha256"]).strip().lower()
    if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != observed_sha:
        raise AuditError(
            f"{location}: sampling_policy_sha256 does not match "
            "sampling_policy_payload"
        )

    return dict(PUBLICATION_SAMPLING_PROTOCOL_METADATA)


def _sampling_metadata_text(value: object) -> str:
    """Return the canonical invariant-summary representation."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _build_method_rng_identity(
    row: Mapping[str, str],
    *,
    method: str,
    config: str,
    location: str,
) -> dict[str, object]:
    """Rebuild the method identity used by the clean comparison runner."""

    common = {
        "attn_implementation": _required_row_text(
            row, column="attn_implementation", location=location
        ),
        "cot": _parse_bool(row.get("cot", ""), column="cot", location=location),
        "dataset_name": _required_row_text(
            row, column="dataset_name", location=location
        ),
        "dataset_sha256": _required_row_text(
            row, column="dataset_sha256", location=location
        ),
        "dtype": _required_row_text(row, column="dtype", location=location),
        "max_new_tokens": _parse_integral_csv_number(
            row.get("max_tokens", ""), column="max_tokens", location=location
        ),
        "model_id": _required_row_text(
            row, column="model_id", location=location
        ),
        "model_revision": _required_row_text(
            row, column="model_revision", location=location
        ),
    }
    protocol_column = f"{method}_protocol"
    protocol = _required_row_text(
        row, column=protocol_column, location=location
    )
    expected_protocol = PUBLICATION_METHOD_PROTOCOLS.get(method)
    if expected_protocol is None:
        raise AuditError(
            f"{location}: strict RNG audit does not support method {method!r}"
        )
    if protocol != expected_protocol:
        raise AuditError(
            f"{location}: {protocol_column} must be "
            f"{expected_protocol!r}, got {protocol!r}"
        )
    sampling_config = (
        _publication_sampling_config(row, location=location)
        if method in GENERATION_METHODS
        else None
    )

    if method == "greedy":
        return {
            "common": common,
            "config": {"do_sample": False, **sampling_config},
            "protocol": protocol,
        }
    if method == "asmc":
        protocol_payload_text = _required_row_text(
            row, column="asmc_protocol_payload", location=location
        )
        try:
            protocol_payload = json.loads(protocol_payload_text)
        except json.JSONDecodeError as exc:
            raise AuditError(
                f"{location}: asmc_protocol_payload is not valid JSON"
            ) from exc
        if not isinstance(protocol_payload, dict):
            raise AuditError(
                f"{location}: asmc_protocol_payload must be a JSON object"
            )
        return {
            "common": common,
            "config_id": config,
            "protocol": protocol,
            "protocol_payload": protocol_payload,
        }
    if method == "naive":
        temperature = _parse_positive_float(
            row.get("temperature", ""),
            column="temperature",
            location=location,
        )
        return {
            "common": common,
            "config": {"temperature": temperature, **sampling_config},
            "protocol": protocol,
        }
    if method == "std":
        return {
            "common": common,
            "config": {"temperature": 1.0, **sampling_config},
            "protocol": protocol,
        }
    if method == "mcmc":
        temperature = _parse_positive_float(
            row.get("temperature", ""),
            column="temperature",
            location=location,
        )
        method_temperature = _parse_positive_float(
            row.get("mcmc_temperature", ""),
            column="mcmc_temperature",
            location=location,
        )
        if not math.isclose(
            temperature, method_temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AuditError(
                f"{location}: mcmc_temperature is inconsistent with temperature"
            )
        return {
            "common": common,
            "config": {
                "blocks": _parse_integral_csv_number(
                    row.get("mcmc_blocks", ""),
                    column="mcmc_blocks",
                    location=location,
                ),
                "steps_per_block": _parse_integral_csv_number(
                    row.get("mcmc_steps", ""),
                    column="mcmc_steps",
                    location=location,
                ),
                "temperature": temperature,
                **sampling_config,
            },
            "protocol": protocol,
        }
    if method == "majority":
        temperature = _parse_positive_float(
            row.get("temperature", ""),
            column="temperature",
            location=location,
        )
        method_temperature = _parse_positive_float(
            row.get("majority_temperature", ""),
            column="majority_temperature",
            location=location,
        )
        if not math.isclose(
            temperature, method_temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AuditError(
                f"{location}: majority_temperature is inconsistent with temperature"
            )
        return {
            "common": common,
            "config": {
                "n_samples": _parse_integral_csv_number(
                    row.get("majority_n", ""),
                    column="majority_n",
                    location=location,
                ),
                "temperature": temperature,
                **sampling_config,
            },
            "protocol": protocol,
        }
    if method == "bestofn":
        chunk_size = _parse_integral_csv_number(
            row.get("bestofn_chunk_size", ""),
            column="bestofn_chunk_size",
            location=location,
        )
        return {
            "common": common,
            "config": {
                "generation_batch_size": chunk_size,
                "length_normalize": True,
                "n": _parse_integral_csv_number(
                    row.get("bestofn_n", ""),
                    column="bestofn_n",
                    location=location,
                ),
                "scoring_batch_size": chunk_size,
                "temperature": _parse_positive_float(
                    row.get("bestofn_temperature", ""),
                    column="bestofn_temperature",
                    location=location,
                ),
                **sampling_config,
            },
            "config_id": config,
            "protocol": protocol,
        }
    raise AuditError(
        f"{location}: strict RNG audit does not support method {method!r}"
    )


def _verify_publication_rng_row(
    row: Mapping[str, str],
    *,
    method: str,
    config: str,
    problem_idx: int,
    location: str,
) -> None:
    """Recompute and validate the runner's per-method isolated RNG key."""

    row_protocol = _required_row_text(
        row, column="rng_protocol", location=location
    )
    method_protocol_column = f"{method}_rng_protocol"
    method_protocol = _required_row_text(
        row, column=method_protocol_column, location=location
    )
    if row_protocol != PUBLICATION_RNG_PROTOCOL:
        raise AuditError(
            f"{location}: rng_protocol must be {PUBLICATION_RNG_PROTOCOL!r}"
        )
    if method_protocol != PUBLICATION_RNG_PROTOCOL:
        raise AuditError(
            f"{location}: {method_protocol_column} must be "
            f"{PUBLICATION_RNG_PROTOCOL!r}"
        )

    base_seed = _parse_integral_csv_number(
        row.get("seed", ""), column="seed", location=location
    )
    method_identity = _build_method_rng_identity(
        row, method=method, config=config, location=location
    )
    key = {
        "base_seed": base_seed,
        "method": method,
        "method_identity": method_identity,
        "problem_idx": problem_idx,
        "rng_protocol": PUBLICATION_RNG_PROTOCOL,
    }
    try:
        expected_payload = json.dumps(
            key,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AuditError(
            f"{location}: RNG key contains a non-canonical value"
        ) from exc
    expected_sha256 = hashlib.sha256(
        expected_payload.encode("utf-8")
    ).hexdigest()
    expected_seed = int(expected_sha256[:8], 16)

    payload_column = f"{method}_rng_key_payload"
    observed_payload = _required_row_text(
        row, column=payload_column, location=location
    )
    if observed_payload != expected_payload:
        raise AuditError(
            f"{location}: {payload_column} does not match the independently "
            "recomputed canonical RNG key"
        )
    sha_column = f"{method}_rng_key_sha256"
    observed_sha256 = _required_row_text(
        row, column=sha_column, location=location
    )
    if not _SHA256_RE.fullmatch(observed_sha256):
        raise AuditError(
            f"{location}: {sha_column} must be a hexadecimal SHA-256"
        )
    if observed_sha256.lower() != expected_sha256:
        raise AuditError(
            f"{location}: {sha_column} does not match {payload_column}"
        )
    seed_column = f"{method}_rng_seed"
    observed_seed = _parse_integral_csv_number(
        row.get(seed_column, ""), column=seed_column, location=location
    )
    if observed_seed != expected_seed:
        raise AuditError(
            f"{location}: {seed_column} does not match the canonical RNG key"
        )


def _parse_problem_idx(value: str, *, location: str) -> int:
    try:
        problem_idx = int(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{location}: invalid problem_idx {value!r}") from exc
    if str(problem_idx) != str(value).strip():
        raise AuditError(f"{location}: problem_idx must be an integer, got {value!r}")
    return problem_idx


def _parse_bool(value: str, *, column: str, location: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise AuditError(f"{location}: {column} must be True/False or 1/0, got {value!r}")


def _parse_nonnegative_float(value: str, *, column: str, location: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{location}: {column} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise AuditError(
            f"{location}: {column} must be finite and non-negative, got {value!r}"
        )
    return parsed


def _parse_positive_float(value: str, *, column: str, location: str) -> float:
    parsed = _parse_nonnegative_float(value, column=column, location=location)
    if parsed <= 0:
        raise AuditError(f"{location}: {column} must be positive, got {value!r}")
    return parsed


def _parse_integral_csv_number(value: str, *, column: str, location: str) -> int:
    """Accept integer text and pandas' lossless ``N.0`` CSV serialization."""

    normalized = str(value).strip()
    try:
        decimal_value = Decimal(normalized)
    except (InvalidOperation, ValueError) as exc:
        raise AuditError(
            f"{location}: {column} must be an integer-valued number"
        ) from exc
    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise AuditError(
            f"{location}: {column} must be an integer-valued number"
        )
    return int(decimal_value)


def _validate_asmc_config_semantics(config: Mapping[str, object]) -> None:
    """Mirror ``ASMCConfig.__post_init__`` for an untrusted JSON payload."""

    location = "asmc_protocol_payload config"
    positive_integer_fields = (
        "n_particles",
        "block_size",
        "max_new_tokens",
        "early_stop_stable_checks",
        "rejuvenation_window",
        "fast_n_particles",
        "hard_n_particles",
    )
    nonnegative_integer_fields = (
        "anneal_tokens",
        "early_stop_min_tokens",
        "min_eos_tokens",
        "prefer_non_eos_until",
        "hard_anneal_tokens",
        "hard_min_eos_tokens",
        "hard_prefer_non_eos_until",
        "hard_early_stop_min_tokens",
    )
    for field in positive_integer_fields + nonnegative_integer_fields:
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AuditError(f"{location} {field} must be an integer")
        if field in positive_integer_fields and value < 1:
            raise AuditError(f"{location} {field} must be positive")
        if field in nonnegative_integer_fields and value < 0:
            raise AuditError(f"{location} {field} must be non-negative")

    if not 1 <= config["fast_n_particles"] <= config["hard_n_particles"]:
        raise AuditError(
            f"{location} adaptive particle counts must satisfy "
            "1 <= fast_n_particles <= hard_n_particles"
        )

    finite_numeric_fields = (
        "alpha_star",
        "ess_threshold",
        "epsilon",
        "alpha_start",
        "early_stop_mass_threshold",
        "early_stop_ess_frac",
        "early_stop_min_parsed_frac",
        "eos_penalty",
        "rejuvenation_fraction",
        "fast_mass_threshold",
        "hard_alpha_start",
        "hard_ess_threshold",
        "hard_eos_penalty",
        "hard_early_stop_mass_threshold",
        "hard_early_stop_ess_frac",
        "hard_early_stop_min_parsed_frac",
    )
    numeric_values: dict[str, float] = {}
    for field in finite_numeric_fields:
        value = config[field]
        # The runner's dataclass stores these fields as Python floats before
        # serializing its content-addressed protocol payload.  Accepting JSON
        # integers here (for example ``4`` in place of ``4.0``) would create a
        # second config ID and RNG stream for the same numerical protocol.
        if not isinstance(value, float):
            raise AuditError(
                f"{location} {field} must use the canonical JSON float type"
            )
        parsed = float(value)
        if not math.isfinite(parsed):
            raise AuditError(f"{location} {field} must be finite")
        if parsed == 0.0 and math.copysign(1.0, parsed) < 0:
            raise AuditError(f"{location} {field} must not use negative zero")
        numeric_values[field] = parsed

    for field in ("alpha_star", "alpha_start", "hard_alpha_start"):
        if numeric_values[field] <= 0:
            raise AuditError(f"{location} {field} must be positive")

    unit_interval_fields = (
        "ess_threshold",
        "early_stop_mass_threshold",
        "early_stop_ess_frac",
        "early_stop_min_parsed_frac",
        "rejuvenation_fraction",
        "fast_mass_threshold",
        "hard_ess_threshold",
        "hard_early_stop_mass_threshold",
        "hard_early_stop_ess_frac",
        "hard_early_stop_min_parsed_frac",
    )
    for field in unit_interval_fields:
        if not 0.0 <= numeric_values[field] <= 1.0:
            raise AuditError(f"{location} {field} must be between 0 and 1")
    if not 0.0 < numeric_values["epsilon"] < 1.0:
        raise AuditError(
            f"{location} epsilon must be strictly between 0 and 1"
        )
    for field in ("eos_penalty", "hard_eos_penalty"):
        if numeric_values[field] < 0:
            raise AuditError(f"{location} {field} must be non-negative")

    c_int_cap = config["c_int_cap"]
    if c_int_cap is not None:
        if not isinstance(c_int_cap, float):
            raise AuditError(
                f"{location} c_int_cap must use the canonical JSON float type"
            )
        if not math.isfinite(c_int_cap) or c_int_cap <= 0:
            raise AuditError(
                f"{location} c_int_cap must be finite and positive when set"
            )
    for field in (
        "use_source_weight",
        "legacy_stop_constraints",
        "enable_rejuvenation",
        "enable_adaptive",
    ):
        if not isinstance(config[field], bool):
            raise AuditError(f"{location} {field} must be boolean")
    if config["anneal_schedule"] not in {"cosine", "linear"}:
        raise AuditError(
            f"{location} anneal_schedule must be 'cosine' or 'linear'"
        )
    if config["stop_token_ids"] != QWEN25_MATH_7B_EOS_TOKEN_IDS:
        raise AuditError(
            f"{location} stop_token_ids must equal the pinned Qwen2.5-Math-7B "
            f"EOS IDs {QWEN25_MATH_7B_EOS_TOKEN_IDS}"
        )


def _validate_asmc_hard_epsilon(value: object) -> None:
    try:
        hard_epsilon = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError("asmc_hard_epsilon must be numeric") from exc
    if not math.isfinite(hard_epsilon) or not math.isclose(
        hard_epsilon,
        ASMC_HARD_EPSILON,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AuditError(
            f"asmc_hard_epsilon must equal {ASMC_HARD_EPSILON}"
        )


def _linear_percentile(values: Sequence[float], percentile: float) -> float:
    """Return a NumPy-compatible linear percentile without requiring NumPy."""

    if not values:
        raise AuditError("cannot compute a percentile for an empty result set")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _metadata_values(
    row: Mapping[str, str], *, method: str, kind: str
) -> Iterable[str]:
    if kind == "method":
        candidates = ("method", f"{method}_method")
    else:
        candidates = ("config", "config_name", f"{method}_config")
    for column in candidates:
        value = row.get(column)
        if value is not None and str(value).strip():
            yield str(value).strip()


def audit_csvs(
    paths: Sequence[str | Path],
    *,
    method: str,
    config: str,
    run_mode: str,
    expected_problem_count: int = 500,
    allow_legacy_aliases: bool = False,
    budget_cap: float | None = None,
    budget_tolerance: float = 1.02,
    require_provenance: bool = False,
) -> dict[str, object]:
    """Audit one configuration assembled from one or more result CSV files.

    Args:
        paths: CSV files whose rows together should cover the benchmark once.
        method: Explicit method label and metric-column prefix (normally
            ``asmc``).
        config: Explicit, human-readable configuration identifier.
        run_mode: ``single`` for baselines, or ``fixed``/``adaptive`` for
            ASMC configurations.
        expected_problem_count: Expected problem IDs are the contiguous range
            ``0 .. expected_problem_count - 1``.
        allow_legacy_aliases: Permit the historical ``*_time`` and
            ``*_total_flops`` columns when the canonical ``*_time_s`` and
            ``*_c_int`` columns are absent.  This is opt-in because older
            ``total_flops`` files may not all share the paper's C_int formula.
        budget_cap: Optional target mean C_int. When supplied, the audit fails
            if realized mean C_int exceeds ``budget_tolerance * budget_cap``.
        budget_tolerance: Multiplicative cap tolerance used by the paper.
        require_provenance: Require complete release metadata on every row and
            reject a dirty code worktree. Legacy result files normally fail
            this publication gate.

    Returns:
        A JSON-serializable summary dictionary.

    Raises:
        AuditError: If any input, schema, value, metadata, coverage, or protocol
            invariant is violated.
    """

    if not paths:
        raise AuditError("at least one input CSV is required")
    if require_provenance and allow_legacy_aliases:
        raise AuditError(
            "publication provenance forbids legacy metric aliases; use the "
            "canonical *_time_s and *_c_int columns"
        )
    method = method.strip().lower()
    config = config.strip()
    run_mode = run_mode.strip().lower()
    if not method:
        raise AuditError("method must not be empty")
    if not config:
        raise AuditError("config must not be empty")
    if run_mode not in {"single", "fixed", "adaptive"}:
        raise AuditError("run_mode must be 'single', 'fixed', or 'adaptive'")
    if method == "asmc" and run_mode not in {"fixed", "adaptive"}:
        raise AuditError("ASMC run_mode must be 'fixed' or 'adaptive'")
    if method != "asmc" and run_mode != "single":
        raise AuditError("non-ASMC run_mode must be 'single'")
    if (
        isinstance(expected_problem_count, bool)
        or not isinstance(expected_problem_count, int)
        or expected_problem_count <= 0
    ):
        raise AuditError("expected_problem_count must be a positive integer")
    if require_provenance and expected_problem_count != 500:
        raise AuditError(
            "publication provenance requires exactly 500 MATH500 problems"
        )
    if budget_cap is not None:
        if (
            isinstance(budget_cap, bool)
            or not isinstance(budget_cap, Real)
            or not math.isfinite(float(budget_cap))
            or budget_cap <= 0
        ):
            raise AuditError("budget_cap must be finite and positive")
    if (
        isinstance(budget_tolerance, bool)
        or not isinstance(budget_tolerance, Real)
        or not math.isfinite(float(budget_tolerance))
        or budget_tolerance < 1.0
    ):
        raise AuditError("budget_tolerance must be finite and at least 1.0")

    correct_column = f"{method}_correct"
    time_column = f"{method}_time_s"
    c_int_column = f"{method}_c_int"
    pass_type_column = f"{method}_pass_type"
    protocol_column = f"{method}_protocol"
    prefill_int_column = f"{method}_prefill_flops"
    decode_int_column = f"{method}_decode_flops"
    total_int_column = f"{method}_total_flops"
    c_tok_column = f"{method}_c_tok"
    c_step_column = f"{method}_c_step"
    n_forward_column = f"{method}_n_forward"
    required_columns = {"problem_idx", correct_column, pass_type_column}
    legacy_time_column = f"{method}_time"
    legacy_c_int_column = f"{method}_total_flops"
    allowed_pass_types = (
        {"fast", "hard"} if run_mode == "adaptive" else {"single"}
    )

    benchmark_rows: list[dict[str, str]] | None = None
    benchmark_artifact: dict[str, object] | None = None
    if require_provenance:
        benchmark_rows, benchmark_artifact = _load_pinned_math500()
        if expected_problem_count > len(benchmark_rows):
            raise AuditError(
                "expected_problem_count exceeds the pinned MATH500 benchmark"
            )

    input_paths = [Path(path) for path in paths]
    seen_locations: dict[int, str] = {}
    duplicate_locations: dict[int, list[str]] = {}
    correctness: list[bool] = []
    times_s: list[float] = []
    c_int_values: list[float] = []
    pass_types: Counter[str] = Counter()
    correctness_evidence: Counter[str] = Counter()
    time_column_usage: Counter[str] = Counter()
    c_int_column_usage: Counter[str] = Counter()
    invariant_columns = PUBLICATION_METADATA_COLUMNS + (
        f"{method}_mode",
        f"{method}_config",
        protocol_column,
        f"{method}_protocol_sha256",
        f"{method}_protocol_payload",
        f"{method}_n",
        f"{method}_temperature",
        f"{method}_chunk_size",
        f"{method}_steps",
        f"{method}_blocks",
        f"{method}_backend",
        f"{method}_use_batched",
        f"{method}_vote_mode",
        f"{method}_use_source_weight",
        f"{method}_c_int_cap",
        f"{method}_legacy_stop_constraints",
        f"{method}_n_particles",
        f"{method}_fast_n_particles",
        f"{method}_hard_n_particles",
        f"{method}_block_size",
        f"{method}_ess_threshold",
        f"{method}_epsilon",
        f"{method}_alpha_start",
        f"{method}_alpha_star",
        f"{method}_anneal_tokens",
        f"{method}_anneal_schedule",
        f"{method}_early_stop_mass_threshold",
        f"{method}_early_stop_min_tokens",
        f"{method}_early_stop_ess_frac",
        f"{method}_early_stop_min_parsed_frac",
        f"{method}_early_stop_stable_checks",
        f"{method}_fast_mass_threshold",
        f"{method}_hard_anneal_tokens",
        f"{method}_hard_alpha_start",
        f"{method}_hard_ess_threshold",
        f"{method}_hard_epsilon",
        f"{method}_hard_early_stop_mass_threshold",
        f"{method}_hard_early_stop_min_tokens",
        f"{method}_hard_early_stop_ess_frac",
        f"{method}_hard_early_stop_min_parsed_frac",
    )
    if method in GENERATION_METHODS:
        invariant_columns += PUBLICATION_SAMPLING_METADATA_COLUMNS
    invariant_metadata: dict[str, str] = {}
    metadata_coverage: Counter[str] = Counter()

    for path in input_paths:
        if not path.is_file():
            raise AuditError(f"input CSV does not exist or is not a file: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AuditError(f"{path}: CSV has no header")
            missing_columns = sorted(required_columns - set(reader.fieldnames))
            if missing_columns:
                raise AuditError(
                    f"{path}: missing required columns: {', '.join(missing_columns)}"
                )
            if require_provenance:
                ground_truth_columns = {
                    "batch_idx",
                    "question",
                    "correct_answer",
                }
                missing_ground_truth_columns = sorted(
                    ground_truth_columns - set(reader.fieldnames)
                )
                if missing_ground_truth_columns:
                    raise AuditError(
                        f"{path}: missing pinned-dataset evidence: "
                        + ", ".join(missing_ground_truth_columns)
                    )
                completion_column = f"{method}_completion"
                if completion_column not in reader.fieldnames:
                    raise AuditError(
                        f"{path}: strict correctness audit requires column "
                        f"{completion_column}"
                    )
                completion_ids_column = f"{method}_completion_token_ids"
                if completion_ids_column not in reader.fieldnames:
                    raise AuditError(
                        f"{path}: strict correctness audit requires column "
                        f"{completion_ids_column}"
                    )
                completion_has_eos_column = f"{method}_completion_has_eos"
                if completion_has_eos_column not in reader.fieldnames:
                    raise AuditError(
                        f"{path}: strict correctness audit requires column "
                        f"{completion_has_eos_column}"
                    )
                rng_columns = {
                    "rng_protocol",
                    f"{method}_rng_protocol",
                    f"{method}_rng_seed",
                    f"{method}_rng_key_sha256",
                    f"{method}_rng_key_payload",
                }
                missing_rng_columns = sorted(
                    rng_columns - set(reader.fieldnames)
                )
                if missing_rng_columns:
                    raise AuditError(
                        f"{path}: missing per-method RNG provenance: "
                        + ", ".join(missing_rng_columns)
                    )
                if method in GENERATION_METHODS:
                    missing_sampling_columns = sorted(
                        set(PUBLICATION_SAMPLING_METADATA_COLUMNS)
                        - set(reader.fieldnames)
                    )
                    if missing_sampling_columns:
                        raise AuditError(
                            f"{path}: missing full-support sampling metadata: "
                            + ", ".join(missing_sampling_columns)
                        )
            if require_provenance and method == "asmc":
                runtime_columns = {
                    "asmc_budget_exhausted",
                    "asmc_budget_exhausted_at_token",
                    "asmc_stop_reason",
                }
                missing_runtime_columns = sorted(
                    runtime_columns - set(reader.fieldnames)
                )
                if missing_runtime_columns:
                    raise AuditError(
                        f"{path}: missing ASMC budget diagnostics: "
                        + ", ".join(missing_runtime_columns)
                    )
            if require_provenance:
                compute_columns = {
                    prefill_int_column,
                    decode_int_column,
                    total_int_column,
                    c_tok_column,
                    c_step_column,
                    n_forward_column,
                }
                missing_compute_columns = sorted(
                    compute_columns - set(reader.fieldnames)
                )
                if missing_compute_columns:
                    raise AuditError(
                        f"{path}: missing canonical compute diagnostics: "
                        + ", ".join(missing_compute_columns)
                    )

            selected_time_column = time_column
            if selected_time_column not in reader.fieldnames:
                if allow_legacy_aliases and legacy_time_column in reader.fieldnames:
                    selected_time_column = legacy_time_column
                else:
                    suffix = (
                        f"; pass --allow-legacy-aliases to accept {legacy_time_column}"
                        if legacy_time_column in reader.fieldnames
                        else ""
                    )
                    raise AuditError(
                        f"{path}: missing required column: {time_column}{suffix}"
                    )

            selected_c_int_column = c_int_column
            if selected_c_int_column not in reader.fieldnames:
                if allow_legacy_aliases and legacy_c_int_column in reader.fieldnames:
                    selected_c_int_column = legacy_c_int_column
                else:
                    suffix = (
                        f"; pass --allow-legacy-aliases only after verifying "
                        f"that {legacy_c_int_column} uses the paper C_int formula"
                        if legacy_c_int_column in reader.fieldnames
                        else ""
                    )
                    raise AuditError(
                        f"{path}: missing required column: {c_int_column}{suffix}"
                    )

            for row in reader:
                location = f"{path}:{reader.line_num}"
                problem_idx = _parse_problem_idx(
                    row.get("problem_idx", ""), location=location
                )
                if problem_idx in seen_locations:
                    duplicate_locations.setdefault(problem_idx, [seen_locations[problem_idx]])
                    duplicate_locations[problem_idx].append(location)
                else:
                    seen_locations[problem_idx] = location

                for observed_method in _metadata_values(
                    row, method=method, kind="method"
                ):
                    if observed_method.lower() != method:
                        raise AuditError(
                            f"{location}: method metadata {observed_method!r} "
                            f"does not match requested method {method!r}"
                        )
                for observed_config in _metadata_values(
                    row, method=method, kind="config"
                ):
                    if observed_config != config:
                        raise AuditError(
                            f"{location}: config metadata {observed_config!r} "
                            f"does not match requested config {config!r}"
                        )

                row_sampling_config = None
                if require_provenance and method in GENERATION_METHODS:
                    row_sampling_config = _publication_sampling_config(
                        row, location=location
                    )

                for column in invariant_columns:
                    if (
                        row_sampling_config is not None
                        and column in PUBLICATION_SAMPLING_METADATA_COLUMNS
                    ):
                        normalized_value = _sampling_metadata_text(
                            row_sampling_config[column]
                        )
                    else:
                        value = row.get(column)
                        if value is None or not str(value).strip():
                            continue
                        normalized_value = str(value).strip()
                    metadata_coverage[column] += 1
                    previous = invariant_metadata.setdefault(column, normalized_value)
                    if normalized_value != previous:
                        raise AuditError(
                            f"{location}: invariant metadata {column}={normalized_value!r} "
                            f"does not match earlier value {previous!r}"
                        )

                dirty_value = row.get("code_git_dirty")
                if dirty_value is not None and str(dirty_value).strip():
                    if str(dirty_value).strip().lower() not in {
                        "true",
                        "false",
                        "1",
                        "0",
                    }:
                        raise AuditError(
                            f"{location}: code_git_dirty must be a boolean, "
                            f"got {dirty_value!r}"
                        )

                observed_mode = row.get(f"{method}_mode")
                if observed_mode is not None and str(observed_mode).strip():
                    if str(observed_mode).strip().lower() != run_mode:
                        raise AuditError(
                            f"{location}: {method}_mode={observed_mode!r} "
                            f"does not match requested mode {run_mode!r}"
                        )

                pass_type = str(row.get(pass_type_column, "")).strip().lower()
                if pass_type not in allowed_pass_types:
                    allowed = ", ".join(sorted(allowed_pass_types))
                    raise AuditError(
                        f"{location}: {run_mode} run requires {pass_type_column} "
                        f"in {{{allowed}}}, got {pass_type!r}"
                    )

                row_correct = _parse_bool(
                    row.get(correct_column, ""),
                    column=correct_column,
                    location=location,
                )
                if require_provenance:
                    assert benchmark_rows is not None
                    if problem_idx < 0 or problem_idx >= len(benchmark_rows):
                        raise AuditError(
                            f"{location}: problem_idx {problem_idx} is outside "
                            "the pinned MATH500 benchmark"
                        )
                    recomputed_correct, evidence_source = (
                        _verify_publication_ground_truth_row(
                            row,
                            method=method,
                            problem_idx=problem_idx,
                            canonical_row=benchmark_rows[problem_idx],
                            location=location,
                        )
                    )
                    if row_correct != recomputed_correct:
                        raise AuditError(
                            f"{location}: {correct_column}={row_correct} does not "
                            f"match repository-recomputed correctness "
                            f"{recomputed_correct}"
                        )
                    _verify_publication_rng_row(
                        row,
                        method=method,
                        config=config,
                        problem_idx=problem_idx,
                        location=location,
                    )
                    row_correct = recomputed_correct
                    correctness_evidence[evidence_source] += 1
                row_time_s = _parse_nonnegative_float(
                    row.get(selected_time_column, ""),
                    column=selected_time_column,
                    location=location,
                )
                row_c_int = _parse_positive_float(
                    row.get(selected_c_int_column, ""),
                    column=selected_c_int_column,
                    location=location,
                )
                if require_provenance:
                    row_c_int_integer = _parse_integral_csv_number(
                        row.get(selected_c_int_column, ""),
                        column=selected_c_int_column,
                        location=location,
                    )
                    prefill_int = _parse_integral_csv_number(
                        row.get(prefill_int_column, ""),
                        column=prefill_int_column,
                        location=location,
                    )
                    decode_int = _parse_integral_csv_number(
                        row.get(decode_int_column, ""),
                        column=decode_int_column,
                        location=location,
                    )
                    total_int = _parse_integral_csv_number(
                        row.get(total_int_column, ""),
                        column=total_int_column,
                        location=location,
                    )
                    c_tok = _parse_integral_csv_number(
                        row.get(c_tok_column, ""),
                        column=c_tok_column,
                        location=location,
                    )
                    c_step = _parse_integral_csv_number(
                        row.get(c_step_column, ""),
                        column=c_step_column,
                        location=location,
                    )
                    n_forward = _parse_integral_csv_number(
                        row.get(n_forward_column, ""),
                        column=n_forward_column,
                        location=location,
                    )
                    if prefill_int < 0 or decode_int < 0:
                        raise AuditError(
                            f"{location}: integrated-attention components must "
                            "be non-negative"
                        )
                    if min(row_c_int_integer, total_int, c_tok, c_step, n_forward) <= 0:
                        raise AuditError(
                            f"{location}: canonical compute totals must be positive"
                        )
                    if prefill_int + decode_int != total_int or total_int != row_c_int_integer:
                        raise AuditError(
                            f"{location}: {prefill_int_column} + "
                            f"{decode_int_column} must equal {total_int_column} "
                            f"and {selected_c_int_column}"
                        )
                    if c_step < n_forward:
                        raise AuditError(
                            f"{location}: {c_step_column} must be at least "
                            f"{n_forward_column}"
                        )
                    if row_time_s <= 0:
                        raise AuditError(
                            f"{location}: publication timing must be positive"
                        )

                if require_provenance and method == "asmc":
                    budget_exhausted = _parse_bool(
                        row.get("asmc_budget_exhausted", ""),
                        column="asmc_budget_exhausted",
                        location=location,
                    )
                    stop_reason = str(row.get("asmc_stop_reason", "")).strip()
                    if stop_reason not in {
                        "early_stop",
                        "max_len",
                        "all_finished",
                        "budget_exhausted",
                    }:
                        raise AuditError(
                            f"{location}: invalid asmc_stop_reason "
                            f"{stop_reason!r}"
                        )
                    exhausted_at = str(
                        row.get("asmc_budget_exhausted_at_token", "")
                    ).strip()
                    cap_text = str(row.get("asmc_c_int_cap", "")).strip().lower()
                    cap_missing = not cap_text
                    if cap_missing or cap_text == "none":
                        cap_value = None
                    else:
                        cap_value = _parse_positive_float(
                            cap_text,
                            column="asmc_c_int_cap",
                            location=location,
                        )

                    if budget_exhausted:
                        if stop_reason != "budget_exhausted":
                            raise AuditError(
                                f"{location}: exhausted budget requires "
                                "asmc_stop_reason='budget_exhausted'"
                            )
                        exhausted_at_int = _parse_integral_csv_number(
                            exhausted_at,
                            column="asmc_budget_exhausted_at_token",
                            location=location,
                        )
                        if exhausted_at_int < -1:
                            raise AuditError(
                                f"{location}: "
                                "asmc_budget_exhausted_at_token must be >= -1"
                            )
                        if cap_value is None:
                            raise AuditError(
                                f"{location}: budget cannot be exhausted when "
                                "asmc_c_int_cap is absent or 'none'"
                            )
                        if row_c_int < cap_value:
                            raise AuditError(
                                f"{location}: budget is marked exhausted but "
                                f"{selected_c_int_column}={row_c_int:g} is below "
                                f"asmc_c_int_cap={cap_value:g}"
                            )
                    else:
                        if stop_reason == "budget_exhausted" or exhausted_at:
                            raise AuditError(
                                f"{location}: non-exhausted budget has "
                                "inconsistent ASMC budget diagnostics"
                            )
                        if (
                            not cap_missing
                            and cap_value is not None
                            and row_c_int >= cap_value
                        ):
                            raise AuditError(
                                f"{location}: {selected_c_int_column}={row_c_int:g} "
                                f"reached/exceeded asmc_c_int_cap={cap_value:g} but "
                                "asmc_budget_exhausted is false"
                            )

                correctness.append(row_correct)
                times_s.append(row_time_s)
                c_int_values.append(row_c_int)
                time_column_usage[selected_time_column] += 1
                c_int_column_usage[selected_c_int_column] += 1
                pass_types[pass_type] += 1

    expected_ids = set(range(expected_problem_count))
    observed_ids = set(seen_locations)
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    coverage_errors: list[str] = []
    if duplicate_locations:
        duplicate_text = ", ".join(
            f"{idx} ({len(locations)} rows)"
            for idx, locations in sorted(duplicate_locations.items())
        )
        coverage_errors.append(f"duplicate problem_idx: {duplicate_text}")
    if missing_ids:
        coverage_errors.append(f"missing problem_idx: {missing_ids}")
    if extra_ids:
        coverage_errors.append(f"unexpected problem_idx: {extra_ids}")
    if coverage_errors:
        raise AuditError("; ".join(coverage_errors))

    required_publication_columns = list(PUBLICATION_METADATA_COLUMNS) + [
        f"{method}_mode",
        f"{method}_config",
        protocol_column,
    ]
    if method in GENERATION_METHODS:
        required_publication_columns.extend(
            PUBLICATION_SAMPLING_METADATA_COLUMNS
        )
    if method == "asmc":
        required_publication_columns.extend(
            [
                "asmc_backend",
                "asmc_protocol_sha256",
                "asmc_protocol_payload",
                "asmc_use_batched",
                "asmc_vote_mode",
                "asmc_use_source_weight",
                "asmc_c_int_cap",
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
            ]
        )
    elif method == "bestofn":
        required_publication_columns.extend(
            [
                "bestofn_n",
                "bestofn_temperature",
                "bestofn_chunk_size",
            ]
        )
    elif method == "mcmc":
        required_publication_columns.extend(
            ["mcmc_steps", "mcmc_blocks", "mcmc_temperature"]
        )
    elif method == "majority":
        required_publication_columns.extend(
            ["majority_n", "majority_temperature"]
        )

    incomplete_publication_columns = sorted(
        column
        for column in required_publication_columns
        if metadata_coverage[column] != len(correctness)
    )
    dirty_normalized = invariant_metadata.get("code_git_dirty", "").lower()
    clean_code = dirty_normalized in {"false", "0"}
    # A summary is publication-complete only after the caller explicitly asks
    # for, and passes, every semantic gate below.  Merely having all columns is
    # not enough to certify an artifact.
    provenance_complete = False
    if require_provenance:
        if incomplete_publication_columns:
            raise AuditError(
                "publication provenance is incomplete for: "
                + ", ".join(incomplete_publication_columns)
            )
        if not clean_code:
            raise AuditError(
                "publication provenance requires code_git_dirty=False"
            )
        dataset_sha = invariant_metadata["dataset_sha256"]
        if not _SHA256_RE.fullmatch(dataset_sha):
            raise AuditError(
                "publication provenance requires dataset_sha256 to be a "
                "64-character hexadecimal SHA-256"
            )
        if invariant_metadata["dataset_name"] != MATH500_DATASET_NAME:
            raise AuditError(
                "publication provenance requires dataset_name="
                f"{MATH500_DATASET_NAME!r}"
            )
        if invariant_metadata["model_id"] != PUBLICATION_MODEL_ID:
            raise AuditError(
                "publication provenance requires model_id="
                f"{PUBLICATION_MODEL_ID!r}"
            )
        if dataset_sha.lower() != MATH500_DATASET_SHA256:
            raise AuditError(
                "publication provenance dataset_sha256 does not match the "
                "pinned repository MATH500 benchmark"
            )
        for column in ("code_git_commit", "model_revision"):
            revision = invariant_metadata[column]
            if not _IMMUTABLE_REVISION_RE.fullmatch(revision):
                raise AuditError(
                    f"publication provenance requires {column} to be an "
                    "immutable 40- or 64-character hexadecimal revision"
                )
        driver_version = invariant_metadata["nvidia_driver_version"].lower()
        if driver_version in {"unknown", "not-applicable", "not-installed"}:
            raise AuditError(
                "publication provenance requires a resolved "
                "nvidia_driver_version"
            )
        if invariant_metadata["attn_implementation"] == "flash_attention_2":
            flash_version = invariant_metadata["flash_attn_version"].lower()
            if flash_version in {"unknown", "not-applicable", "not-installed"}:
                raise AuditError(
                    "flash_attention_2 publication runs require a resolved "
                    "flash_attn_version"
                )

        compute_schema = invariant_metadata["compute_schema"]
        if compute_schema != PUBLICATION_COMPUTE_SCHEMA:
            raise AuditError(
                "publication provenance requires compute_schema="
                f"{PUBLICATION_COMPUTE_SCHEMA!r}, got {compute_schema!r}"
            )
        timing_schema = invariant_metadata["timing_schema"]
        if timing_schema != PUBLICATION_TIMING_SCHEMA:
            raise AuditError(
                "publication provenance requires timing_schema="
                f"{PUBLICATION_TIMING_SCHEMA!r}, got {timing_schema!r}"
            )

        if method == "asmc":
            protocol_sha = invariant_metadata["asmc_protocol_sha256"]
            if not _SHA256_RE.fullmatch(protocol_sha):
                raise AuditError(
                    "publication provenance requires a hexadecimal "
                    "asmc_protocol_sha256"
                )
            protocol_payload_text = invariant_metadata["asmc_protocol_payload"]
            observed_protocol_sha = hashlib.sha256(
                protocol_payload_text.encode("utf-8")
            ).hexdigest()
            if observed_protocol_sha.lower() != protocol_sha.lower():
                raise AuditError(
                    "asmc_protocol_sha256 does not match asmc_protocol_payload"
                )
            try:
                protocol_payload = json.loads(protocol_payload_text)
            except json.JSONDecodeError as exc:
                raise AuditError("asmc_protocol_payload is not valid JSON") from exc
            if not isinstance(protocol_payload, dict) or not isinstance(
                protocol_payload.get("config"), dict
            ):
                raise AuditError(
                    "asmc_protocol_payload must contain a configuration object"
                )
            try:
                canonical_protocol_payload_text = json.dumps(
                    protocol_payload,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise AuditError(
                    "asmc_protocol_payload cannot be serialized as canonical JSON"
                ) from exc
            if protocol_payload_text != canonical_protocol_payload_text:
                raise AuditError(
                    "asmc_protocol_payload must use the runner's canonical "
                    "sorted compact JSON representation"
                )
            expected_payload_fields = {
                "backend": invariant_metadata["asmc_backend"],
                "cot": _parse_bool(
                    invariant_metadata["cot"],
                    column="cot",
                    location="publication provenance",
                ),
                "vote_mode": invariant_metadata["asmc_vote_mode"],
            }
            for field, expected in expected_payload_fields.items():
                if protocol_payload.get(field) != expected:
                    raise AuditError(
                        f"asmc_protocol_payload {field} is inconsistent with "
                        "row provenance"
                    )

            payload_config = protocol_payload["config"]
            expected_config_keys = {
                "alpha_star",
                "c_int_cap",
                "n_particles",
                "block_size",
                "max_new_tokens",
                "ess_threshold",
                "epsilon",
                "anneal_tokens",
                "alpha_start",
                "anneal_schedule",
                "use_source_weight",
                "early_stop_mass_threshold",
                "early_stop_min_tokens",
                "early_stop_ess_frac",
                "early_stop_min_parsed_frac",
                "early_stop_stable_checks",
                "legacy_stop_constraints",
                "min_eos_tokens",
                "prefer_non_eos_until",
                "eos_penalty",
                "stop_token_ids",
                "enable_rejuvenation",
                "rejuvenation_fraction",
                "rejuvenation_window",
                "enable_adaptive",
                "fast_mass_threshold",
                "fast_n_particles",
                "hard_n_particles",
                "hard_anneal_tokens",
                "hard_alpha_start",
                "hard_ess_threshold",
                "hard_min_eos_tokens",
                "hard_prefer_non_eos_until",
                "hard_eos_penalty",
                "hard_early_stop_min_tokens",
                "hard_early_stop_mass_threshold",
                "hard_early_stop_ess_frac",
                "hard_early_stop_min_parsed_frac",
            }
            if set(payload_config) != expected_config_keys:
                raise AuditError(
                    "asmc_protocol_payload config keys do not match the "
                    "canonical ASMCConfig schema"
                )
            _validate_asmc_config_semantics(payload_config)

            expected_config_id = (
                f"asmc-{run_mode}-n{payload_config['n_particles']}-"
                f"{protocol_payload['vote_mode']}-{protocol_sha[:16].lower()}"
            )
            if invariant_metadata["asmc_config"] != expected_config_id:
                raise AuditError(
                    "asmc_config must equal the content-addressed runner ID "
                    f"{expected_config_id!r}"
                )
            if config != expected_config_id:
                raise AuditError(
                    "requested config must equal the content-addressed "
                    f"runner ID {expected_config_id!r}"
                )

            integer_metadata_map = {
                "n_particles": "asmc_n_particles",
                "block_size": "asmc_block_size",
                "max_new_tokens": "max_tokens",
                "anneal_tokens": "asmc_anneal_tokens",
                "early_stop_min_tokens": "asmc_early_stop_min_tokens",
                "early_stop_stable_checks": "asmc_early_stop_stable_checks",
                "fast_n_particles": "asmc_fast_n_particles",
                "hard_n_particles": "asmc_hard_n_particles",
                "hard_anneal_tokens": "asmc_hard_anneal_tokens",
                "hard_early_stop_min_tokens": "asmc_hard_early_stop_min_tokens",
            }
            for config_field, metadata_field in integer_metadata_map.items():
                metadata_value = _parse_integral_csv_number(
                    invariant_metadata[metadata_field],
                    column=metadata_field,
                    location="publication provenance",
                )
                if payload_config[config_field] != metadata_value:
                    raise AuditError(
                        f"asmc_protocol_payload {config_field} is inconsistent "
                        "with row provenance"
                    )

            numeric_metadata_map = {
                "alpha_star": "asmc_alpha_star",
                "ess_threshold": "asmc_ess_threshold",
                "epsilon": "asmc_epsilon",
                "alpha_start": "asmc_alpha_start",
                "early_stop_mass_threshold": "asmc_early_stop_mass_threshold",
                "early_stop_ess_frac": "asmc_early_stop_ess_frac",
                "early_stop_min_parsed_frac": "asmc_early_stop_min_parsed_frac",
                "fast_mass_threshold": "asmc_fast_mass_threshold",
                "hard_alpha_start": "asmc_hard_alpha_start",
                "hard_ess_threshold": "asmc_hard_ess_threshold",
                "hard_early_stop_mass_threshold": (
                    "asmc_hard_early_stop_mass_threshold"
                ),
                "hard_early_stop_ess_frac": "asmc_hard_early_stop_ess_frac",
                "hard_early_stop_min_parsed_frac": (
                    "asmc_hard_early_stop_min_parsed_frac"
                ),
            }
            for config_field, metadata_field in numeric_metadata_map.items():
                try:
                    payload_value = float(payload_config[config_field])
                    metadata_value = float(invariant_metadata[metadata_field])
                except (TypeError, ValueError) as exc:
                    raise AuditError(
                        f"asmc_protocol_payload {config_field} must be numeric"
                    ) from exc
                if not math.isfinite(payload_value) or not math.isclose(
                    payload_value,
                    metadata_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise AuditError(
                        f"asmc_protocol_payload {config_field} is inconsistent "
                        "with row provenance"
                    )

            if payload_config["anneal_schedule"] != invariant_metadata[
                "asmc_anneal_schedule"
            ]:
                raise AuditError(
                    "asmc_protocol_payload anneal_schedule is inconsistent "
                    "with row provenance"
                )
            for config_field, metadata_field in (
                ("use_source_weight", "asmc_use_source_weight"),
                ("legacy_stop_constraints", "asmc_legacy_stop_constraints"),
            ):
                expected_bool = _parse_bool(
                    invariant_metadata[metadata_field],
                    column=metadata_field,
                    location="publication provenance",
                )
                if payload_config[config_field] is not expected_bool:
                    raise AuditError(
                        f"asmc_protocol_payload {config_field} is inconsistent "
                        "with row provenance"
                    )
            if payload_config["enable_adaptive"] is not (run_mode == "adaptive"):
                raise AuditError(
                    "asmc_protocol_payload enable_adaptive is inconsistent "
                    "with run_mode"
                )
            cap_text = invariant_metadata["asmc_c_int_cap"].lower()
            expected_cap = None if cap_text == "none" else float(cap_text)
            if payload_config["c_int_cap"] != expected_cap:
                raise AuditError(
                    "asmc_protocol_payload c_int_cap is inconsistent with row provenance"
                )
            _validate_asmc_hard_epsilon(
                invariant_metadata["asmc_hard_epsilon"]
            )
            expected_internal_defaults = {
                "min_eos_tokens": 128,
                "prefer_non_eos_until": 512,
                "eos_penalty": 5.0,
                "enable_rejuvenation": False,
                "rejuvenation_fraction": 0.25,
                "rejuvenation_window": 96,
                "hard_min_eos_tokens": 256,
                "hard_prefer_non_eos_until": 768,
                "hard_eos_penalty": 6.0,
            }
            for field, expected in expected_internal_defaults.items():
                if payload_config[field] != expected:
                    raise AuditError(
                        f"asmc_protocol_payload {field} does not match the "
                        "corrected runner default"
                    )

            vote_mode = invariant_metadata["asmc_vote_mode"]
            if vote_mode not in {
                "weighted",
                "weighted_no_source",
                "majority",
                "majority_no_source",
            }:
                raise AuditError(
                    "publication provenance contains an unknown asmc_vote_mode"
                )
            expected_source_weight = vote_mode in {"weighted", "majority"}
            observed_source_weight = _parse_bool(
                invariant_metadata["asmc_use_source_weight"],
                column="asmc_use_source_weight",
                location="publication provenance",
            )
            if observed_source_weight != expected_source_weight:
                raise AuditError(
                    "asmc_use_source_weight is inconsistent with asmc_vote_mode"
                )

            backend = invariant_metadata["asmc_backend"]
            if backend not in {"batched", "sequential"}:
                raise AuditError(
                    "publication provenance requires asmc_backend to be "
                    "'batched' or 'sequential'"
                )
            use_batched = _parse_bool(
                invariant_metadata["asmc_use_batched"],
                column="asmc_use_batched",
                location="publication provenance",
            )
            if use_batched != (backend == "batched"):
                raise AuditError(
                    "asmc_use_batched is inconsistent with asmc_backend"
                )
            if backend != "batched":
                raise AuditError(
                    "publication ASMC provenance requires the cache-coherent "
                    "batched backend"
                )

            cap = invariant_metadata["asmc_c_int_cap"].lower()
            if run_mode == "adaptive" and cap == "none":
                raise AuditError(
                    "publication adaptive ASMC requires a finite positive "
                    "asmc_c_int_cap"
                )
            if cap != "none":
                _parse_positive_float(
                    cap,
                    column="asmc_c_int_cap",
                    location="publication provenance",
                )

            if invariant_metadata["asmc_anneal_schedule"] not in {
                "cosine",
                "linear",
            }:
                raise AuditError(
                    "publication provenance contains an unknown "
                    "asmc_anneal_schedule"
                )

        _parse_bool(
            invariant_metadata["cot"],
            column="cot",
            location="publication provenance",
        )
        provenance_complete = True

    mean_c_int = fmean(c_int_values)
    budget_limit = (
        budget_cap * budget_tolerance if budget_cap is not None else None
    )
    if budget_limit is not None and mean_c_int > budget_limit:
        raise AuditError(
            f"mean C_int {mean_c_int:.6g} exceeds budget limit "
            f"{budget_limit:.6g} ({budget_tolerance:.4g} x {budget_cap:.6g})"
        )

    return {
        "schema_version": 1,
        "method": method,
        "config": config,
        "run_mode": run_mode,
        "n_files": len(input_paths),
        "n_rows": len(correctness),
        "n_problems": len(observed_ids),
        "accuracy": fmean(correctness),
        "time_p50_s": _linear_percentile(times_s, 0.50),
        "time_p95_s": _linear_percentile(times_s, 0.95),
        "mean_c_int": mean_c_int,
        "budget_cap": budget_cap,
        "budget_tolerance": budget_tolerance if budget_cap is not None else None,
        "budget_limit": budget_limit,
        "pass_type_counts": dict(sorted(pass_types.items())),
        "metric_column_usage": {
            "time": dict(sorted(time_column_usage.items())),
            "c_int": dict(sorted(c_int_column_usage.items())),
        },
        "invariant_metadata": dict(sorted(invariant_metadata.items())),
        "metadata_coverage": dict(sorted(metadata_coverage.items())),
        "legacy_aliases_allowed": allow_legacy_aliases,
        "publication_provenance_required": require_provenance,
        "provenance_complete": provenance_complete,
        "correctness_recomputed": require_provenance,
        "rng_keys_recomputed": require_provenance,
        "correctness_evidence_counts": dict(
            sorted(correctness_evidence.items())
        ),
        "benchmark_artifact": benchmark_artifact,
        "incomplete_publication_columns": incomplete_publication_columns,
        "source_files": [str(path) for path in input_paths],
        "source_artifacts": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in input_paths
        ],
    }


def write_json_summary(summary: Mapping[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(output_path)


def write_csv_summary(summary: Mapping[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(summary)
    for key, value in list(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary_path.replace(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit complete, non-overlapping per-problem result CSVs and compute "
            "accuracy, latency percentiles, and mean C_int."
        )
    )
    parser.add_argument("csv", nargs="+", type=Path, help="one or more result CSVs")
    parser.add_argument(
        "--method",
        default="asmc",
        help="method label and metric-column prefix (default: asmc)",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help=(
            "require complete release metadata on every row and a clean Git "
            "worktree; legacy shards normally fail this gate"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="explicit configuration identifier; never inferred from a path",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("single", "fixed", "adaptive"),
        help=(
            "ASMC fixed requires pass_type=single; ASMC adaptive permits "
            "fast/hard; non-ASMC methods use single"
        ),
    )
    parser.add_argument(
        "--expected-problems",
        type=int,
        default=500,
        help="require exactly problem_idx 0..N-1 (default: 500)",
    )
    parser.add_argument(
        "--allow-legacy-aliases",
        action="store_true",
        help=(
            "accept *_time and *_total_flops when canonical columns are absent; "
            "use only after confirming *_total_flops follows the paper C_int formula"
        ),
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        help="optional target mean C_int to enforce",
    )
    parser.add_argument(
        "--budget-tolerance",
        type=float,
        default=1.02,
        help="allowed multiplier on --budget-cap (default: 1.02)",
    )
    parser.add_argument("--json-out", type=Path, help="write summary as JSON")
    parser.add_argument("--csv-out", type=Path, help="write one-row summary as CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        input_paths = {path.resolve() for path in args.csv}
        dependency_paths = set(input_paths)
        if args.require_provenance:
            repository_root = Path(__file__).resolve().parents[1]
            dependency_paths.update(
                {
                    MATH500_PATH.resolve(),
                    Path(__file__).resolve(),
                    (repository_root / "grader_utils" / "math_grader.py").resolve(),
                    (repository_root / "grader_utils" / "parse_utils.py").resolve(),
                }
            )
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
            raise AuditError(
                "summary output paths and their atomic temporary paths must "
                "all be distinct"
            )
        if dependency_paths.intersection(output_write_paths):
            raise AuditError(
                "an output or atomic temporary path must not overwrite an input "
                "CSV, pinned benchmark, audit source, or grader source"
            )
        dependency_hashes = (
            {path: _sha256_file(path) for path in dependency_paths}
            if args.require_provenance
            else {}
        )
        summary = audit_csvs(
            args.csv,
            method=args.method,
            config=args.config,
            run_mode=args.mode,
            expected_problem_count=args.expected_problems,
            allow_legacy_aliases=args.allow_legacy_aliases,
            budget_cap=args.budget_cap,
            budget_tolerance=args.budget_tolerance,
            require_provenance=args.require_provenance,
        )
        if args.json_out:
            write_json_summary(summary, args.json_out)
        if args.csv_out:
            write_csv_summary(summary, args.csv_out)
        for path in output_paths:
            if not path.is_file():
                raise AuditError(f"summary output was not created: {path}")
        for path, expected_sha256 in dependency_hashes.items():
            if not path.is_file() or _sha256_file(path) != expected_sha256:
                raise AuditError(
                    f"publication input changed while writing outputs: {path}"
                )
    except (AuditError, OSError) as exc:
        print(f"result audit failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
