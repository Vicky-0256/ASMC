"""Batched Best-of-N sampling used by the paper baselines.

The implementation follows the baseline described in the paper:

1. draw ``n`` independent completions in generation batches;
2. rescore every sampled token sequence (including terminal EOS) with
   micro-batched teacher forcing; and
3. select the completion with the largest length-normalized sequence
   log-probability.

Both generation and scoring reduce their batch size after a CUDA out-of-memory
error.  Other runtime errors are deliberately not hidden.  When compute
tracking is enabled, the shared model-forward instrumentation covers generation
and teacher-forcing calls, so callers do not need to wrap the model themselves.

This module only depends on PyTorch and the small public compute-accounting
utilities in this repository.  It does not import an experiment runner, a
dataset, or a task-specific answer parser.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

import torch
import torch.nn.functional as F

try:  # Package import: ``from ASMC.bestofn import ...``
    from .compute_instrumentation import instrument_model
    from .compute_tracker import ComputeTracker
except ImportError:  # Script-style import with ASMC/ on ``sys.path``.
    from compute_instrumentation import instrument_model
    from compute_tracker import ComputeTracker


_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


# Neutralize every Hugging Face sampling warper that can remove or penalize
# tokens.  Explicit kwargs take precedence over a model repository's bundled
# ``generation_config.json`` and therefore keep candidate generation on the
# full temperature-scaled vocabulary.  The comparison runner imports this
# same definition for all other stochastic ``generate`` paths and provenance.
FULL_SUPPORT_SAMPLING_KWARGS: Dict[str, Any] = {
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
SAMPLING_POLICY = "full-support-temperature-only-v1"
SAMPLING_PROTOCOL_METADATA: Dict[str, Any] = {
    "sampling_policy": SAMPLING_POLICY,
    **{
        f"sampling_{name}": "none" if value is None else value
        for name, value in FULL_SUPPORT_SAMPLING_KWARGS.items()
    },
}


def _resolved_eos_and_pad_token_ids(tokenizer: Any) -> Tuple[List[int], int]:
    """Resolve an explicit, canonical termination policy from a tokenizer."""
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(raw_eos, set):
        eos_token_ids = sorted(int(token_id) for token_id in raw_eos)
    elif isinstance(raw_eos, (list, tuple)):
        eos_token_ids = [int(token_id) for token_id in raw_eos]
    elif raw_eos is None:
        eos_token_ids = []
    else:
        eos_token_ids = [int(raw_eos)]
    eos_token_ids = list(dict.fromkeys(eos_token_ids))
    if not eos_token_ids:
        raise ValueError("tokenizer must define at least one eos_token_id")

    raw_pad = getattr(tokenizer, "pad_token_id", None)
    pad_token_id = eos_token_ids[0] if raw_pad is None else int(raw_pad)
    return eos_token_ids, pad_token_id


def resolved_sampling_generation_kwargs(tokenizer: Any) -> Dict[str, Any]:
    """Return the complete explicit kwargs for full-support generation."""
    eos_token_ids, pad_token_id = _resolved_eos_and_pad_token_ids(tokenizer)
    return {
        **FULL_SUPPORT_SAMPLING_KWARGS,
        "eos_token_id": eos_token_ids,
        "pad_token_id": pad_token_id,
    }


def resolved_sampling_protocol_metadata(tokenizer: Any) -> Dict[str, Any]:
    """Return lossless row/RNG metadata for the resolved sampling policy."""
    resolved_kwargs = resolved_sampling_generation_kwargs(tokenizer)
    payload = json.dumps(
        resolved_kwargs,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        **SAMPLING_PROTOCOL_METADATA,
        "sampling_eos_token_ids": json.dumps(
            resolved_kwargs["eos_token_id"],
            separators=(",", ":"),
        ),
        "sampling_pad_token_id": resolved_kwargs["pad_token_id"],
        "sampling_policy_payload": payload,
        "sampling_policy_sha256": hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
    }


@dataclass
class Candidate:
    """One generated completion, excluding its prompt but retaining terminal EOS.

    ``tokens`` is the complete sampled sequence used by the Best-of-N scoring
    rule.  Consequently, a terminal EOS contributes both its teacher-forced
    log-probability and one token to the length-normalisation denominator.
    ``text`` is decoded with special tokens skipped and is intended only for
    downstream answer parsing and display.
    """

    tokens: torch.Tensor
    text: str
    has_eos: bool
    hit_limit: bool

    @property
    def length(self) -> int:
        """Number of sampled tokens used for teacher-forcing scoring."""

        return int(self.tokens.numel())


@dataclass(frozen=True)
class AdaptiveBatchStats:
    """Diagnostics for an adaptively micro-batched phase."""

    initial_chunk_size: int
    final_chunk_size: int
    oom_retries: int

    @property
    def used_oom_fallback(self) -> bool:
        return self.oom_retries > 0


@dataclass
class BestOfNResult:
    """Completion, scores, diagnostics, and optional compute measurements."""

    completion: str
    best_index: int
    best_score: float
    candidates: List[Candidate]
    scores: List[float]
    generation_stats: AdaptiveBatchStats
    scoring_stats: AdaptiveBatchStats
    compute: Optional[Dict[str, Any]] = None

    def to_info_dict(self) -> Dict[str, Any]:
        """Return flat metadata suitable for a CSV/JSON experiment record."""

        finite_scores = [score for score in self.scores if math.isfinite(score)]
        lengths = [candidate.length for candidate in self.candidates]
        info: Dict[str, Any] = {
            "n": len(self.candidates),
            "best_idx": self.best_index,
            "best_score": self.best_score,
            "best_len": self.candidates[self.best_index].length,
            "score_mean": (
                sum(finite_scores) / len(finite_scores) if finite_scores else None
            ),
            "len_mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "generation_chunk_initial": self.generation_stats.initial_chunk_size,
            "generation_chunk_used": self.generation_stats.final_chunk_size,
            "generation_oom_retries": self.generation_stats.oom_retries,
            "generation_oom_fallback": (
                self.generation_stats.used_oom_fallback
            ),
            "score_chunk_initial": self.scoring_stats.initial_chunk_size,
            "score_chunk_used": self.scoring_stats.final_chunk_size,
            "score_oom_retries": self.scoring_stats.oom_retries,
            "score_oom_fallback": self.scoring_stats.used_oom_fallback,
        }
        if self.compute is not None:
            info["compute"] = dict(self.compute)
        return info


def next_smaller_chunk_size(chunk_size: int) -> int:
    """Return the next OOM retry size by halving, with a lower bound of one."""

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be positive")
    return max(1, chunk_size // 2)


def is_out_of_memory_error(error: BaseException) -> bool:
    """Return whether ``error`` represents a CUDA out-of-memory condition."""

    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(error, cuda_oom_type):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def normalize_sequence_log_probability(
    total_log_probability: float,
    sequence_length: int,
    *,
    enabled: bool = True,
) -> float:
    """Apply the paper's sequence-length normalization to one log-probability."""

    if sequence_length < 1:
        return float("-inf")
    if enabled:
        return float(total_log_probability) / sequence_length
    return float(total_log_probability)


def select_best_candidate(scores: Sequence[float]) -> int:
    """Return the stable argmax of candidate scores (the first index wins ties)."""

    if not scores:
        raise ValueError("at least one candidate score is required")
    if any(math.isnan(float(score)) for score in scores):
        raise ValueError("candidate scores must not contain NaN")
    return max(range(len(scores)), key=lambda index: scores[index])


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _map_in_adaptive_chunks(
    items: Sequence[_Item],
    initial_chunk_size: int,
    process_batch: Callable[[Sequence[_Item]], List[_Result]],
    *,
    phase: str,
) -> Tuple[List[_Result], AdaptiveBatchStats]:
    """Map batches while persistently reducing the chunk cap after CUDA OOM."""

    if (
        isinstance(initial_chunk_size, bool)
        or not isinstance(initial_chunk_size, int)
        or initial_chunk_size < 1
    ):
        raise ValueError("chunk size must be positive")

    outputs: List[_Result] = []
    offset = 0
    chunk_size = initial_chunk_size
    oom_retries = 0

    while offset < len(items):
        attempted_size = min(chunk_size, len(items) - offset)
        batch = items[offset : offset + attempted_size]
        try:
            batch_outputs = process_batch(batch)
        except Exception as error:
            if not is_out_of_memory_error(error):
                raise
            oom_retries += 1
            _clear_cuda_cache()
            if attempted_size == 1:
                raise RuntimeError(
                    f"Best-of-N {phase} ran out of memory at chunk_size=1"
                ) from error
            # Base the reduction on the batch that actually failed.  This also
            # guarantees progress when the final, partial batch is the one that
            # exhausts memory.
            chunk_size = next_smaller_chunk_size(attempted_size)
            continue

        if len(batch_outputs) != attempted_size:
            raise RuntimeError(
                f"{phase} returned {len(batch_outputs)} outputs for "
                f"a batch of {attempted_size}"
            )
        outputs.extend(batch_outputs)
        offset += attempted_size

    return outputs, AdaptiveBatchStats(
        initial_chunk_size=initial_chunk_size,
        final_chunk_size=chunk_size,
        oom_retries=oom_retries,
    )


def _as_token_id_set(value: Any) -> Set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(token_id) for token_id in value}
    return {int(value)}


def _first_matching_position(tokens: torch.Tensor, token_ids: Set[int]) -> Optional[int]:
    if not token_ids:
        return None
    for position, token_id in enumerate(tokens.detach().cpu().tolist()):
        if int(token_id) in token_ids:
            return position
    return None


def _resolve_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_ids = _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
    if eos_ids:
        return min(eos_ids)
    # Token zero is only used for masked teacher-forcing padding.  It cannot
    # affect a causal score because the corresponding attention-mask entries
    # are zero.
    return 0


def _candidates_from_sequences(
    sequences: torch.Tensor,
    tokenizer: Any,
    *,
    prompt_length: int,
    max_new_tokens: int,
) -> List[Candidate]:
    """Convert decoder-only ``generate`` output into completion candidates."""

    eos_ids = _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    # Preserve an int or ordered list exactly as exposed by the tokenizer;
    # both forms are accepted by current Transformers generation APIs.
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_ids = {int(pad_token_id)} if pad_token_id is not None else set()

    candidates: List[Candidate] = []
    for sequence in sequences:
        if sequence.numel() < prompt_length:
            raise RuntimeError("model.generate returned a sequence shorter than the prompt")
        raw_completion = sequence[prompt_length:]
        eos_position = _first_matching_position(raw_completion, eos_ids)
        has_eos = eos_position is not None

        if eos_position is not None:
            # EOS is part of the sampled sequence.  Retaining it here is
            # essential: Best-of-N ranks complete unconditional generations,
            # so p(EOS | prefix) and EOS's contribution to the sequence-length
            # denominator must both be included.  Tokens after the first EOS
            # are generation padding and are not part of the candidate.
            end = eos_position + 1
        else:
            # A distinct padding token can follow a completion stopped by a
            # custom stopping criterion.  If pad == EOS, the EOS branch above
            # handles it without losing the termination flag.
            distinct_pad_ids = pad_ids - eos_ids
            pad_position = _first_matching_position(
                raw_completion, distinct_pad_ids
            )
            end = pad_position if pad_position is not None else raw_completion.numel()

        completion_tokens = raw_completion[:end].clone()
        hit_limit = not has_eos and int(end) >= max_new_tokens
        text = tokenizer.decode(
            completion_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
        )
        candidates.append(
            Candidate(
                tokens=completion_tokens,
                text=text,
                has_eos=has_eos,
                hit_limit=hit_limit,
            )
        )
    return candidates


@torch.no_grad()
def generate_candidates(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    chunk_size: int = 8,
) -> Tuple[List[Candidate], AdaptiveBatchStats]:
    """Draw ``n`` independent completions with batched Hugging Face generation."""

    if prompt_ids.ndim != 1 or prompt_ids.numel() < 1:
        raise ValueError("prompt_ids must be a non-empty one-dimensional tensor")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be positive")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be positive")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive for Best-of-N sampling")

    generation_policy = resolved_sampling_generation_kwargs(tokenizer)
    pad_token_id = generation_policy["pad_token_id"]

    def generate_batch(batch_markers: Sequence[int]) -> List[Candidate]:
        batch_size = len(batch_markers)
        input_ids = prompt_ids.unsqueeze(0).repeat(batch_size, 1)
        generation_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "return_dict_in_generate": True,
            "output_scores": False,
            **generation_policy,
        }

        generated = model.generate(**generation_kwargs)
        sequences = getattr(generated, "sequences", None)
        if sequences is None:
            raise RuntimeError("model.generate must return an object with .sequences")
        if sequences.ndim != 2 or sequences.shape[0] != batch_size:
            raise RuntimeError(
                "model.generate returned an unexpected number or shape of sequences"
            )
        return _candidates_from_sequences(
            sequences,
            tokenizer,
            prompt_length=prompt_ids.numel(),
            max_new_tokens=max_new_tokens,
        )

    markers = list(range(n))
    return _map_in_adaptive_chunks(
        markers,
        chunk_size,
        generate_batch,
        phase="generation",
    )


def _extract_logits(model_output: Any) -> torch.Tensor:
    logits = getattr(model_output, "logits", None)
    if logits is None and isinstance(model_output, (tuple, list)) and model_output:
        logits = model_output[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("model forward must return logits with shape [B, S, V]")
    return logits


@torch.no_grad()
def score_candidates(
    model: Any,
    prompt_ids: torch.Tensor,
    candidates: Sequence[Candidate],
    *,
    chunk_size: int = 8,
    pad_token_id: int = 0,
    length_normalize: bool = True,
) -> Tuple[List[float], AdaptiveBatchStats]:
    """Teacher-force and score candidates with adaptive OOM micro-batching."""

    if prompt_ids.ndim != 1 or prompt_ids.numel() < 1:
        raise ValueError("prompt_ids must be a non-empty one-dimensional tensor")

    scores = [float("-inf")] * len(candidates)
    valid_indices = [
        index for index, candidate in enumerate(candidates) if candidate.length > 0
    ]
    prompt_length = int(prompt_ids.numel())
    device = prompt_ids.device

    def score_batch(batch_indices: Sequence[int]) -> List[Tuple[int, float]]:
        batch_candidates = [candidates[index] for index in batch_indices]
        max_completion_length = max(candidate.length for candidate in batch_candidates)
        total_length = prompt_length + max_completion_length
        batch_input = torch.full(
            (len(batch_indices), total_length),
            int(pad_token_id),
            dtype=prompt_ids.dtype,
            device=device,
        )
        attention_mask = torch.zeros_like(batch_input)

        for row, candidate in enumerate(batch_candidates):
            candidate_tokens = candidate.tokens.to(device=device, dtype=prompt_ids.dtype)
            candidate_end = prompt_length + candidate.length
            batch_input[row, :prompt_length] = prompt_ids
            batch_input[row, prompt_length:candidate_end] = candidate_tokens
            attention_mask[row, :candidate_end] = 1

        output = model(
            input_ids=batch_input,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = _extract_logits(output)

        batch_scores: List[Tuple[int, float]] = []
        for row, (original_index, candidate) in enumerate(
            zip(batch_indices, batch_candidates)
        ):
            prediction_logits = logits[
                row,
                prompt_length - 1 : prompt_length + candidate.length - 1,
                :,
            ]
            target_tokens = candidate.tokens.to(device=device, dtype=torch.long)
            token_log_probabilities = F.log_softmax(
                prediction_logits.float(), dim=-1
            ).gather(1, target_tokens.unsqueeze(1)).squeeze(1)
            total_log_probability = float(token_log_probabilities.sum().item())
            score = normalize_sequence_log_probability(
                total_log_probability,
                candidate.length,
                enabled=length_normalize,
            )
            batch_scores.append((int(original_index), score))
        return batch_scores

    scored_pairs, stats = _map_in_adaptive_chunks(
        valid_indices,
        chunk_size,
        score_batch,
        phase="teacher-forcing scoring",
    )
    for index, score in scored_pairs:
        scores[index] = score
    return scores, stats


@torch.no_grad()
def sample_best_of_n(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    generation_batch_size: int = 8,
    scoring_batch_size: Optional[int] = None,
    length_normalize: bool = True,
    track_compute: bool = True,
    compute_tracker: Optional[ComputeTracker] = None,
    sync_cuda: bool = True,
) -> BestOfNResult:
    """Run the complete paper Best-of-N baseline for one prompt.

    ``compute_tracker``, when supplied, is reset before this run.  The function
    installs the repository's shared forward hook itself; callers must not wrap
    the model in a second instrumentation context.  Set ``track_compute=False``
    to disable both the hook and timing.
    """

    if scoring_batch_size is None:
        scoring_batch_size = generation_batch_size
    for field, value in (
        ("generation_batch_size", generation_batch_size),
        ("scoring_batch_size", scoring_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not track_compute and compute_tracker is not None:
        raise ValueError("compute_tracker requires track_compute=True")

    tracker: Optional[ComputeTracker]
    if track_compute:
        tracker = compute_tracker if compute_tracker is not None else ComputeTracker()
        tracker.reset()
        tracking_context = instrument_model(model, tracker, sync_cuda=sync_cuda)
    else:
        tracker = None
        tracking_context = nullcontext()

    with tracking_context:
        candidates, generation_stats = generate_candidates(
            model,
            tokenizer,
            prompt_ids,
            n=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chunk_size=generation_batch_size,
        )
        scores, scoring_stats = score_candidates(
            model,
            prompt_ids,
            candidates,
            chunk_size=scoring_batch_size,
            pad_token_id=_resolve_pad_token_id(tokenizer),
            length_normalize=length_normalize,
        )
        best_index = select_best_candidate(scores)

    compute = tracker.get_stats() if tracker is not None else None
    return BestOfNResult(
        completion=candidates[best_index].text,
        best_index=best_index,
        best_score=scores[best_index],
        candidates=candidates,
        scores=scores,
        generation_stats=generation_stats,
        scoring_stats=scoring_stats,
        compute=compute,
    )


__all__ = [
    "AdaptiveBatchStats",
    "BestOfNResult",
    "Candidate",
    "generate_candidates",
    "is_out_of_memory_error",
    "next_smaller_chunk_size",
    "normalize_sequence_log_probability",
    "sample_best_of_n",
    "score_candidates",
    "select_best_candidate",
]
