"""Compute accounting for cached autoregressive inference.

The paper reports three complementary proxy metrics:

``C_int``
    Integrated attention positions.  A prefill of length ``L`` costs
    ``B * L * (L + 1) / 2``.  A cached forward with ``S`` query tokens and
    ``P`` cached tokens costs ``B * (S * P + S * (S + 1) / 2)``.
``C_tok``
    Number of input/query tokens processed, including the prompt.
``C_step``
    Sum of the active batch size over forward calls, ``sum_k B_k``.

``forward_calls`` is deliberately kept separate from ``C_step``.  The legacy
``*_flops`` keys remain available for old result-writing code, but they are
aliases for the corresponding ``C_int`` components rather than hardware FLOPs.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterable, List, Optional


def _normalise_lengths(
    lengths: Iterable[int], expected_batch_size: int
) -> List[int]:
    """Return validated Python integer lengths for a padded batch."""
    if hasattr(lengths, "detach"):
        lengths = lengths.detach().cpu().tolist()
    else:
        lengths = list(lengths)

    if len(lengths) != expected_batch_size:
        raise ValueError(
            "effective_lens must contain one length per batch element "
            f"({len(lengths)} != {expected_batch_size})"
        )

    normalised = [int(length) for length in lengths]
    if any(length < 0 for length in normalised):
        raise ValueError("sequence lengths must be non-negative")
    return normalised


@dataclass
class ComputeTracker:
    """Accumulate model-independent inference compute proxies."""

    # Integrated-attention components.  The old names are retained for output
    # compatibility with the first public experiment runner.
    prefill_flops: int = 0
    decode_flops: int = 0

    # Forward-call counts, kept distinct from C_step = sum of batch sizes.
    n_prefill: int = 0
    n_decode: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0

    # Token counts (C_tok components).
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0

    # Synchronous wall-clock time.  CUDA synchronisation, when wanted, is
    # handled by compute_instrumentation rather than this data container.
    total_time: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        """Reset all counters for a new problem or method."""
        self.prefill_flops = 0
        self.decode_flops = 0
        self.n_prefill = 0
        self.n_decode = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.total_time = 0.0
        self._start_time = 0.0

    def start_timer(self) -> None:
        """Start the synchronous wall-clock timer."""
        self._start_time = time.perf_counter()

    def stop_timer(self) -> None:
        """Stop the wall-clock timer and accumulate elapsed time."""
        if self._start_time > 0:
            self.total_time += time.perf_counter() - self._start_time
            self._start_time = 0.0

    def log_prefill(
        self,
        batch_size: int,
        seq_len: int,
        effective_lens: Optional[Iterable[int]] = None,
    ) -> None:
        """Log a full-attention prefill.

        ``effective_lens`` should be supplied for a padded batch.  Its entries
        are the non-padding prompt lengths, so padding does not inflate either
        ``C_int`` or ``C_tok``.
        """
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")

        if effective_lens is None:
            lengths = [seq_len] * batch_size
        else:
            lengths = _normalise_lengths(effective_lens, batch_size)
            if any(length > seq_len for length in lengths):
                raise ValueError("an effective prefill length exceeds seq_len")

        self.prefill_flops += sum(
            length * (length + 1) // 2 for length in lengths
        )
        self.total_prefill_tokens += sum(lengths)
        self.prefill_steps += batch_size
        self.n_prefill += 1

    def log_decode(
        self,
        batch_size: int,
        total_len: Optional[int] = None,
        step_tokens: int = 1,
        *,
        past_len: Optional[int] = None,
        effective_lens: Optional[Iterable[int]] = None,
    ) -> None:
        """Log a cached forward containing one or more query tokens.

        Args:
            batch_size: Number of active sequences in the forward call.
            total_len: Sequence length after adding the query tokens.  Kept as
                the second positional argument for backward compatibility.
            step_tokens: Number of query tokens ``S`` in this forward call.
            past_len: Cached length ``P`` before this call.  Specify either
                ``past_len`` or ``total_len``; if both are given they must agree.
            effective_lens: Optional per-sample total lengths after this call,
                useful when the batch has padding or heterogeneous histories.
        """
        batch_size = int(batch_size)
        step_tokens = int(step_tokens)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if step_tokens < 1:
            raise ValueError("step_tokens must be positive")

        if past_len is not None:
            past_len = int(past_len)
            if past_len < 0:
                raise ValueError("past_len must be non-negative")
            inferred_total_len = past_len + step_tokens
            if total_len is not None and int(total_len) != inferred_total_len:
                raise ValueError("total_len must equal past_len + step_tokens")
            total_len = inferred_total_len
        elif total_len is None:
            raise ValueError("one of total_len or past_len must be provided")
        else:
            total_len = int(total_len)
            past_len = total_len - step_tokens
            if past_len < 0:
                raise ValueError("total_len must be at least step_tokens")

        if effective_lens is None:
            total_lengths = [total_len] * batch_size
        else:
            total_lengths = _normalise_lengths(effective_lens, batch_size)
            if any(length < step_tokens for length in total_lengths):
                raise ValueError(
                    "each effective decode length must include all query tokens"
                )

        # For each sequence, query token i attends to P+i positions (i=1..S).
        triangular_query = step_tokens * (step_tokens + 1) // 2
        self.decode_flops += sum(
            step_tokens * (length - step_tokens) + triangular_query
            for length in total_lengths
        )
        self.total_decode_tokens += batch_size * step_tokens
        self.decode_steps += batch_size
        self.n_decode += 1

    @property
    def C_int(self) -> int:
        """Integrated-attention cost."""
        return self.prefill_flops + self.decode_flops

    @property
    def C_tok(self) -> int:
        """Total prompt and query tokens processed."""
        return self.total_prefill_tokens + self.total_decode_tokens

    @property
    def C_step(self) -> int:
        """Step cost from the paper: sum of active batch sizes over calls."""
        return self.prefill_steps + self.decode_steps

    @property
    def forward_calls(self) -> int:
        """Literal number of calls to model.forward()."""
        return self.n_prefill + self.n_decode

    def get_metrics(self) -> Dict[str, Any]:
        """Return the canonical paper metrics and their components."""
        return {
            "C_int": self.C_int,
            "C_tok": self.C_tok,
            "C_step": self.C_step,
            "forward_calls": self.forward_calls,
            "n_prefill": self.n_prefill,
            "n_decode": self.n_decode,
            "prefill_int": self.prefill_flops,
            "decode_int": self.decode_flops,
            "prefill_tok": self.total_prefill_tokens,
            "decode_tok": self.total_decode_tokens,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return canonical metrics plus legacy result-writer keys."""
        total_flops = self.C_int
        return {
            **self.get_metrics(),
            # Backward-compatible names.  These are attention-position proxies,
            # not estimates of hardware floating-point operations.
            "prefill_flops": self.prefill_flops,
            "decode_flops": self.decode_flops,
            "total_flops": total_flops,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "total_tokens": self.C_tok,
            "total_time": self.total_time,
            "prefill_ratio": (
                self.prefill_flops / total_flops if total_flops else 0.0
            ),
            "decode_ratio": (
                self.decode_flops / total_flops if total_flops else 0.0
            ),
            "tokens_per_second": (
                self.C_tok / self.total_time if self.total_time > 0 else 0.0
            ),
        }

    def __str__(self) -> str:
        return (
            "ComputeTracker("
            f"C_int={self.C_int:,}, C_tok={self.C_tok:,}, "
            f"C_step={self.C_step:,}, forward_calls={self.forward_calls}, "
            f"time={self.total_time:.2f}s)"
        )


_global_tracker: Optional[ComputeTracker] = None


def get_global_tracker() -> ComputeTracker:
    """Get the global compute tracker, creating it on first use."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ComputeTracker()
    return _global_tracker


def set_global_tracker(tracker: ComputeTracker) -> None:
    """Set the global compute tracker."""
    global _global_tracker
    _global_tracker = tracker


def reset_global_tracker() -> None:
    """Reset the global tracker, creating it if necessary."""
    get_global_tracker().reset()
