"""
Compute Tracker for LLM Inference

Tracks computational cost of LLM inference:
- Prefill: O(B * S^2) - quadratic in sequence length
- Decode: O(B * L) - linear in past_len + current tokens

Usage:
    tracker = ComputeTracker()
    tracker.reset()
    # ... run inference ...
    stats = tracker.get_stats()
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import time


@dataclass
class ComputeTracker:
    """
    Tracks computational cost of LLM inference.
    
    Prefill cost: B * S^2 (quadratic - full attention over new tokens)
    Decode cost: B * L (linear - attention over past + 1 new token)
    
    Where:
        B = batch size
        S = sequence length (for prefill)
        L = past_len + new_tokens (for decode)
    """
    
    # Cumulative compute costs
    prefill_flops: int = 0      # Sum of B * S^2 over all prefill calls
    decode_flops: int = 0       # Sum of B * L over all decode calls
    
    # Call counts (for debugging)
    n_prefill: int = 0          # Number of prefill calls
    n_decode: int = 0           # Number of decode calls
    
    # Token counts
    total_prefill_tokens: int = 0   # Total tokens processed in prefill
    total_decode_tokens: int = 0    # Total tokens generated in decode
    
    # Timing (wall clock)
    total_time: float = 0.0
    _start_time: float = field(default=0.0, repr=False)
    
    def reset(self):
        """Reset all counters for a new problem/method."""
        self.prefill_flops = 0
        self.decode_flops = 0
        self.n_prefill = 0
        self.n_decode = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.total_time = 0.0
        self._start_time = 0.0
    
    def start_timer(self):
        """Start wall clock timer."""
        self._start_time = time.perf_counter()
    
    def stop_timer(self):
        """Stop wall clock timer and accumulate."""
        if self._start_time > 0:
            self.total_time += time.perf_counter() - self._start_time
            self._start_time = 0.0
    
    def log_prefill(self, batch_size: int, seq_len: int):
        """
        Log a prefill operation.
        
        Prefill computes full self-attention over seq_len tokens.
        Cost: O(B * S^2) where S = seq_len
        
        Args:
            batch_size: Number of sequences in batch (B)
            seq_len: Length of input sequence (S)
        """
        cost = batch_size * seq_len * seq_len
        self.prefill_flops += cost
        self.n_prefill += 1
        self.total_prefill_tokens += batch_size * seq_len
    
    def log_decode(self, batch_size: int, total_len: int):
        """
        Log a decode operation.
        
        Decode computes attention from 1 new token to all past tokens.
        Cost: O(B * L) where L = past_len + 1
        
        Args:
            batch_size: Number of sequences in batch (B)
            total_len: Total sequence length including new token (L = past_len + 1)
        """
        cost = batch_size * total_len
        self.decode_flops += cost
        self.n_decode += 1
        self.total_decode_tokens += batch_size  # 1 new token per sequence
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all tracked statistics.
        
        Returns:
            Dict with all compute metrics
        """
        total_flops = self.prefill_flops + self.decode_flops
        
        return {
            # Primary metrics
            "prefill_flops": self.prefill_flops,
            "decode_flops": self.decode_flops,
            "total_flops": total_flops,
            
            # Call counts
            "n_prefill": self.n_prefill,
            "n_decode": self.n_decode,
            
            # Token counts
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "total_tokens": self.total_prefill_tokens + self.total_decode_tokens,
            
            # Timing
            "total_time": self.total_time,
            
            # Derived metrics
            "prefill_ratio": self.prefill_flops / total_flops if total_flops > 0 else 0.0,
            "decode_ratio": self.decode_flops / total_flops if total_flops > 0 else 0.0,
            "tokens_per_second": (self.total_prefill_tokens + self.total_decode_tokens) / self.total_time if self.total_time > 0 else 0.0,
        }
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"ComputeTracker(\n"
            f"  prefill: {stats['prefill_flops']:,} flops ({stats['n_prefill']} calls, {stats['total_prefill_tokens']} tokens)\n"
            f"  decode:  {stats['decode_flops']:,} flops ({stats['n_decode']} calls, {stats['total_decode_tokens']} tokens)\n"
            f"  total:   {stats['total_flops']:,} flops\n"
            f"  time:    {stats['total_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)\n"
            f")"
        )


# Global tracker instance (can be overridden per-method)
_global_tracker: ComputeTracker = None


def get_global_tracker() -> ComputeTracker:
    """Get the global compute tracker, creating one if needed."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ComputeTracker()
    return _global_tracker


def set_global_tracker(tracker: ComputeTracker):
    """Set the global compute tracker."""
    global _global_tracker
    _global_tracker = tracker


def reset_global_tracker():
    """Reset the global compute tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
    else:
        _global_tracker = ComputeTracker()
