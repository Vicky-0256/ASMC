#!/usr/bin/env python3
"""Benchmark one cache-resampling event with CUDA events.

The two measured implementations have the same endpoint: a resampled batch
whose KV cache has advanced by one token.

``rebuild``
    Gather the selected token prefixes, replay the complete prefix to rebuild
    the KV cache, then perform one cached decode step.

``reorder``
    Gather every layer of the existing KV cache by ancestor index, then
    perform the same cached decode step.

Ancestor sampling and the initial prefix prefill happen outside the timed
region. CUDA events measure GPU work only; explicit synchronization brackets
each event. The script deliberately has no top-level Torch or Transformers
imports, so ``--help`` and argument validation work on CPU-only machines.
"""

import argparse
import csv
import gc
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
GIB = 1024 ** 3
EVENT_DEFINITION = (
    "apply ancestor indices, restore a decodable KV cache, and advance the "
    "resampled batch by one token"
)


def _git_state() -> Dict[str, Optional[Any]]:
    """Return actual repository provenance without asserting a clean state."""

    repository = Path(__file__).resolve().parents[1]
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _cuda_device(value: str) -> str:
    if not re.fullmatch(r"cuda(?::\d+)?", value):
        raise argparse.ArgumentTypeError(
            "must be a CUDA device such as 'cuda' or 'cuda:0'"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure prefix replay (rebuild) versus KV-cache ancestor gather "
            "(reorder) for one resampling event. CUDA is only accessed after "
            "arguments have been parsed."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-7B",
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision; use an immutable commit hash for paper runs",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model and KV-cache dtype",
    )
    parser.add_argument(
        "--device",
        type=_cuda_device,
        default="cuda",
        help="CUDA device used for events, for example cuda:0",
    )
    parser.add_argument(
        "--N",
        "--n",
        dest="n_values",
        metavar="N",
        type=_positive_int,
        nargs="+",
        default=[16],
        help="Particle count(s) to benchmark (default: 16)",
    )
    parser.add_argument(
        "--L",
        "--l",
        dest="l_values",
        metavar="L",
        type=_positive_int,
        nargs="+",
        default=[512],
        help="Prefix length(s) to benchmark (default: 512)",
    )
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=2,
        help="Untimed events per point (default: 2)",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=10,
        help="Timed events per point (default: 10)",
    )
    parser.add_argument(
        "--implementations",
        choices=("reorder", "rebuild"),
        nargs="+",
        default=["reorder", "rebuild"],
        help="Implementations to run (default: both)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Optional Transformers attention backend",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact the Hugging Face Hub",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repository code (disabled by default)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("cache_resampling_results.json"),
        help="Structured output path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("cache_resampling_results.csv"),
        help="Flat per-point output path",
    )
    return parser


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        "%s_median" % prefix: statistics.median(values),
        "%s_mean" % prefix: statistics.fmean(values),
        "%s_std" % prefix: statistics.pstdev(values),
        "%s_p95" % prefix: _percentile(values, 95.0),
        "%s_min" % prefix: min(values),
        "%s_max" % prefix: max(values),
    }


def _serialize_cli_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "device": args.device,
        "N": list(args.n_values),
        "L": list(args.l_values),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "implementations": list(dict.fromkeys(args.implementations)),
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }


def _base_document(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "benchmark": "cache_resampling_single_event",
        "code": _git_state(),
        "event_definition": EVENT_DEFINITION,
        "timing": {
            "clock": "torch.cuda.Event",
            "synchronization": "explicit before and after each event",
            "includes_cpu_dispatch": False,
            "ancestor_sampling_timed": False,
            "initial_prefill_timed": False,
            "rebuild": "prefix gather + full prefix replay + one cached decode",
            "reorder": "KV-cache ancestor gather + one cached decode",
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "configuration": _serialize_cli_config(args),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "results": [],
    }


def _json_safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _flatten_for_csv(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    common: Dict[str, Any] = {
        "schema_version": document["schema_version"],
        "benchmark": document["benchmark"],
        "event_definition": document["event_definition"],
        "started_at_utc": document["started_at_utc"],
        "completed_at_utc": document["completed_at_utc"],
    }
    for prefix in ("code", "configuration", "environment"):
        for key, value in document[prefix].items():
            common["%s_%s" % (prefix, key)] = _json_safe_scalar(value)

    rows = []
    for result in document["results"]:
        row = dict(common)
        row.update({key: _json_safe_scalar(value) for key, value in result.items()})
        rows.append(row)
    return rows


def _write_outputs(document: Dict[str, Any], args: argparse.Namespace) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary_json.replace(args.output_json)

    rows = _flatten_for_csv(document)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    temporary_csv.replace(args.output_csv)


def _resolve_device(torch: Any, requested: str) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; this benchmark requires CUDA events, but "
            "--help and argument validation remain available without a GPU"
        )
    device = torch.device(requested)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            "requested %s, but %d CUDA device(s) are visible"
            % (requested, torch.cuda.device_count())
        )
    torch.cuda.set_device(index)
    return torch.device("cuda:%d" % index)


def _dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _decoder_backbone(model: Any) -> Tuple[Any, str]:
    """Return a decoder without the LM head to avoid full-prefix logits."""
    for attribute in ("model", "transformer", "gpt_neox"):
        candidate = getattr(model, attribute, None)
        if candidate is not None and candidate is not model and callable(candidate):
            return candidate, attribute
    raise RuntimeError(
        "cannot identify a decoder backbone without an LM head for model class "
        "%s; supported layouts expose .model, .transformer, or .gpt_neox"
        % type(model).__name__
    )


def _extract_cache(outputs: Any) -> Any:
    cache = getattr(outputs, "past_key_values", None)
    if cache is None:
        raise RuntimeError("model forward did not return past_key_values")
    return cache


def _reorder_cache(cache: Any, ancestors: Any) -> Tuple[Any, str]:
    """Gather every KV-cache layer along its batch/particle dimension."""
    if hasattr(cache, "reorder_cache"):
        reordered = cache.reorder_cache(ancestors)
        return (cache if reordered is None else reordered), "cache.reorder_cache"

    if hasattr(cache, "batch_select_indices"):
        reordered = cache.batch_select_indices(ancestors)
        return (
            cache if reordered is None else reordered,
            "cache.batch_select_indices",
        )

    if isinstance(cache, (tuple, list)):
        reordered_layers = []
        for layer_index, layer in enumerate(cache):
            if not isinstance(layer, (tuple, list)):
                raise TypeError(
                    "legacy cache layer %d is not a tuple/list" % layer_index
                )
            reordered_tensors = []
            for tensor in layer:
                if tensor is None:
                    reordered_tensors.append(None)
                elif tensor.ndim == 0 or tensor.shape[0] != ancestors.numel():
                    raise ValueError(
                        "legacy cache tensor has no particle batch dimension: %s"
                        % (tuple(tensor.shape),)
                    )
                else:
                    reordered_tensors.append(tensor.index_select(0, ancestors))
            reordered_layers.append(tuple(reordered_tensors))
        return tuple(reordered_layers), "legacy_tuple.index_select"

    raise TypeError("unsupported cache type: %s" % type(cache).__name__)


def _empty_result(
    n_particles: Optional[int],
    prefix_length: Optional[int],
    implementation: str,
    status: str,
    phase: str,
    error: BaseException,
) -> Dict[str, Any]:
    return {
        "N": n_particles,
        "L": prefix_length,
        "implementation": implementation,
        "status": status,
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": " ".join(str(error).split())[:1000],
        "samples": 0,
    }


def _benchmark_point(
    torch: Any,
    backbone: Any,
    model: Any,
    device: Any,
    n_particles: int,
    prefix_length: int,
    implementation: str,
    warmup: int,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    point_seed = seed + n_particles * 1000003 + prefix_length * 9176
    generator = torch.Generator(device=device)
    generator.manual_seed(point_seed)

    vocab_size = int(model.config.vocab_size)
    if vocab_size < 2:
        raise RuntimeError("model vocabulary must contain at least two tokens")

    input_ids = torch.randint(
        0,
        vocab_size,
        (n_particles, prefix_length),
        generator=generator,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    decode_tokens = torch.ones(
        (n_particles, 1), dtype=torch.long, device=device
    )
    decode_attention_mask = torch.ones(
        (n_particles, prefix_length + 1), dtype=torch.long, device=device
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies_ms: List[float] = []
    peak_allocated_gib: List[float] = []
    event_delta_gib: List[float] = []
    unique_ancestor_ratios: List[float] = []
    cache_type = None
    gather_implementation = None

    with torch.inference_mode():
        for iteration in range(warmup + repeats):
            ancestors = torch.randint(
                0,
                n_particles,
                (n_particles,),
                generator=generator,
                dtype=torch.long,
                device=device,
            )
            unique_ratio = torch.unique(ancestors).numel() / float(n_particles)

            # Establish the same pre-resampling state for both implementations.
            initial_outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            initial_cache = _extract_cache(initial_outputs)
            cache_type = "%s.%s" % (
                type(initial_cache).__module__,
                type(initial_cache).__name__,
            )
            del initial_outputs

            if implementation == "rebuild":
                # Prefix replay discards the old cache before measurement.
                del initial_cache

            torch.cuda.synchronize(device)
            memory_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event.record()

            if implementation == "rebuild":
                selected_ids = input_ids.index_select(0, ancestors)
                selected_mask = attention_mask.index_select(0, ancestors)
                replay_outputs = backbone(
                    input_ids=selected_ids,
                    attention_mask=selected_mask,
                    use_cache=True,
                    return_dict=True,
                )
                event_cache = _extract_cache(replay_outputs)
                del replay_outputs
                resumed_outputs = backbone(
                    input_ids=decode_tokens,
                    attention_mask=decode_attention_mask,
                    past_key_values=event_cache,
                    use_cache=True,
                    return_dict=True,
                )
                gather_implementation = "prefix.index_select+prefix_replay"
            else:
                event_cache, gather_implementation = _reorder_cache(
                    initial_cache, ancestors
                )
                resumed_outputs = backbone(
                    input_ids=decode_tokens,
                    attention_mask=decode_attention_mask,
                    past_key_values=event_cache,
                    use_cache=True,
                    return_dict=True,
                )

            # Force access to the endpoint cache before ending the event.
            resumed_cache = _extract_cache(resumed_outputs)
            if resumed_cache is None:  # pragma: no cover - defensive clarity
                raise RuntimeError("resampling event did not produce a cache")
            end_event.record()
            torch.cuda.synchronize(device)

            elapsed_ms = float(start_event.elapsed_time(end_event))
            peak_bytes = torch.cuda.max_memory_allocated(device)
            if iteration >= warmup:
                latencies_ms.append(elapsed_ms)
                peak_allocated_gib.append(peak_bytes / float(GIB))
                event_delta_gib.append(
                    max(0, peak_bytes - memory_before) / float(GIB)
                )
                unique_ancestor_ratios.append(unique_ratio)

            del resumed_outputs, resumed_cache, event_cache, ancestors
            if implementation == "rebuild":
                del selected_ids, selected_mask
            else:
                del initial_cache

    result: Dict[str, Any] = {
        "N": n_particles,
        "L": prefix_length,
        "implementation": implementation,
        "status": "ok",
        "phase": "complete",
        "error_type": None,
        "error_message": None,
        "samples": len(latencies_ms),
        "warmup": warmup,
        "repeats": repeats,
        "cache_type": cache_type,
        "gather_implementation": gather_implementation,
        "ancestor_unique_ratio_mean": statistics.fmean(unique_ancestor_ratios),
    }
    result.update(_summarize(latencies_ms, "event_ms"))
    result.update(_summarize(peak_allocated_gib, "peak_allocated_gib"))
    result.update(_summarize(event_delta_gib, "event_delta_gib"))
    return result


def _load_dependencies() -> Tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "install the benchmark dependencies (torch and transformers) "
            "before running a CUDA benchmark"
        ) from error
    return torch, transformers


def run(args: argparse.Namespace) -> int:
    document = _base_document(args)
    try:
        torch, transformers = _load_dependencies()
        device = _resolve_device(torch, args.device)
    except Exception as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    properties = torch.cuda.get_device_properties(device)
    document["environment"].update(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": properties.name,
            "gpu_total_memory_gib": properties.total_memory / float(GIB),
            "gpu_compute_capability": "%d.%d"
            % (properties.major, properties.minor),
        }
    )

    load_kwargs: Dict[str, Any] = {
        "revision": args.revision,
        "torch_dtype": _dtype_from_name(torch, args.dtype),
        "device_map": {"": str(device)},
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, **load_kwargs
        )
        model.eval()
        backbone, backbone_attribute = _decoder_backbone(model)
        document["environment"].update(
            {
                "model_class": type(model).__name__,
                "backbone_attribute": backbone_attribute,
                "resolved_model_revision": getattr(
                    model.config, "_commit_hash", None
                ),
            }
        )
    except Exception as error:
        status = (
            "oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            else "error"
        )
        document["results"].append(
            _empty_result(None, None, "model_load", status, "model_load", error)
        )
        document["status"] = "failed"
        document["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_outputs(document, args)
        print("model loading failed: %s" % error, file=sys.stderr)
        return 1

    implementations = list(dict.fromkeys(args.implementations))
    for prefix_length in args.l_values:
        for n_particles in args.n_values:
            for implementation in implementations:
                print(
                    "[%s] N=%d L=%d" % (
                        implementation,
                        n_particles,
                        prefix_length,
                    ),
                    flush=True,
                )
                try:
                    result = _benchmark_point(
                        torch=torch,
                        backbone=backbone,
                        model=model,
                        device=device,
                        n_particles=n_particles,
                        prefix_length=prefix_length,
                        implementation=implementation,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        seed=args.seed,
                    )
                    print(
                        "  median %.3f ms; peak %.3f GiB"
                        % (
                            result["event_ms_median"],
                            result["peak_allocated_gib_median"],
                        )
                    )
                except Exception as error:
                    status = (
                        "oom"
                        if isinstance(error, torch.cuda.OutOfMemoryError)
                        else "error"
                    )
                    result = _empty_result(
                        n_particles,
                        prefix_length,
                        implementation,
                        status,
                        "setup_or_event",
                        error,
                    )
                    print("  %s: %s" % (status.upper(), error), file=sys.stderr)
                    gc.collect()
                    torch.cuda.empty_cache()

                document["results"].append(result)
                document["completed_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                # Checkpoint after every point so a later OOM does not lose data.
                _write_outputs(document, args)

    failed_points = [
        result for result in document["results"] if result.get("status") != "ok"
    ]
    document["status"] = "complete" if not failed_points else "completed_with_errors"
    document["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_outputs(document, args)
    return 0 if not failed_points else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_json.resolve() == args.output_csv.resolve():
        parser.error("--output-json and --output-csv must be different paths")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
