"""
Compute Instrumentation for LLM Inference

Wraps model.forward() to automatically log compute costs to a tracker.
Works with any HuggingFace model that uses past_key_values for KV caching.

Usage:
    tracker = ComputeTracker()
    with instrument_model(model, tracker):
        # All forward calls are automatically tracked
        output = model.generate(...)
    
    stats = tracker.get_stats()
"""

import torch
from contextlib import contextmanager
from typing import Optional, Callable, Any
from functools import wraps

from compute_tracker import ComputeTracker, get_global_tracker


def _get_batch_size_and_seq_len(input_ids: Optional[torch.Tensor], 
                                  inputs_embeds: Optional[torch.Tensor]) -> tuple:
    """
    Extract batch size and sequence length from input.
    
    Args:
        input_ids: Token IDs tensor [B, S]
        inputs_embeds: Embedding tensor [B, S, D]
    
    Returns:
        (batch_size, seq_len)
    """
    if input_ids is not None:
        return input_ids.shape[0], input_ids.shape[1]
    elif inputs_embeds is not None:
        return inputs_embeds.shape[0], inputs_embeds.shape[1]
    else:
        return 1, 0


def _get_past_len(past_key_values) -> int:
    """
    Get the length of cached past key values.
    
    Args:
        past_key_values: KV cache (tuple of tuples, DynamicCache, or None)
    
    Returns:
        past_len: Number of cached tokens
    """
    if past_key_values is None:
        return 0
    
    # Handle DynamicCache (modern transformers)
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values.get_seq_length()
    
    # Handle tuple format (older transformers)
    # past_key_values[layer_idx] = (key, value)
    # key shape: [batch, num_heads, seq_len, head_dim]
    try:
        if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            if isinstance(first_layer, tuple) and len(first_layer) >= 1:
                key_tensor = first_layer[0]
                if hasattr(key_tensor, 'shape') and len(key_tensor.shape) >= 3:
                    return key_tensor.shape[2]  # seq_len dimension
    except (IndexError, AttributeError, TypeError):
        pass
    
    return 0


def create_wrapped_forward(original_forward: Callable, 
                           tracker: ComputeTracker,
                           sync_cuda: bool = True) -> Callable:
    """
    Create a wrapped forward function that logs compute costs.
    
    Args:
        original_forward: The original model.forward method
        tracker: ComputeTracker instance to log to
        sync_cuda: Whether to sync CUDA before timing (recommended for accurate timing)
    
    Returns:
        Wrapped forward function
    """
    @wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        # Extract input_ids from args or kwargs
        input_ids = None
        inputs_embeds = None
        past_key_values = None
        
        # Check kwargs first (more common in modern usage)
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif 'inputs_embeds' in kwargs:
            inputs_embeds = kwargs['inputs_embeds']
        elif len(args) > 0:
            # First positional arg is usually input_ids
            input_ids = args[0]
        
        if 'past_key_values' in kwargs:
            past_key_values = kwargs['past_key_values']
        
        # Get dimensions
        batch_size, seq_len = _get_batch_size_and_seq_len(input_ids, inputs_embeds)
        past_len = _get_past_len(past_key_values)
        
        # Log compute cost
        if past_key_values is None or past_len == 0:
            # Prefill: no KV cache, processing fresh sequence
            # Cost: O(B * S^2)
            if seq_len > 0:
                tracker.log_prefill(batch_size, seq_len)
        else:
            # Decode: using KV cache, typically generating 1 token at a time
            # Cost: O(B * (past_len + seq_len))
            total_len = past_len + seq_len
            tracker.log_decode(batch_size, total_len)
        
        # Call original forward
        return original_forward(*args, **kwargs)
    
    return wrapped_forward


@contextmanager
def instrument_model(model: torch.nn.Module, 
                     tracker: Optional[ComputeTracker] = None,
                     sync_cuda: bool = True):
    """
    Context manager to instrument a model's forward pass for compute tracking.
    
    All calls to model.forward() within the context will be logged to the tracker.
    This works with model.generate() since it internally calls forward().
    
    Args:
        model: HuggingFace model (or any model with .forward method)
        tracker: ComputeTracker instance (uses global tracker if None)
        sync_cuda: Whether to sync CUDA for accurate timing
    
    Yields:
        The tracker being used
    
    Example:
        tracker = ComputeTracker()
        with instrument_model(model, tracker):
            output = model.generate(input_ids, max_new_tokens=100)
        print(tracker.get_stats())
    """
    if tracker is None:
        tracker = get_global_tracker()
    
    # Save original forward
    original_forward = model.forward
    
    # Create wrapped forward
    wrapped_forward = create_wrapped_forward(original_forward, tracker, sync_cuda)
    
    # CUDA sync helper
    def sync():
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    try:
        # Replace forward
        model.forward = wrapped_forward
        
        # Start timer with CUDA sync
        sync()
        tracker.start_timer()
        
        yield tracker
        
    finally:
        # Stop timer with CUDA sync
        sync()
        tracker.stop_timer()
        
        # Restore original forward
        model.forward = original_forward


class ModelInstrumenter:
    """
    Class-based instrumenter for more control over when tracking starts/stops.
    
    Usage:
        instrumenter = ModelInstrumenter(model)
        
        # Method 1: Context manager
        with instrumenter.track() as tracker:
            model.generate(...)
        
        # Method 2: Manual start/stop
        tracker = instrumenter.start()
        model.generate(...)
        stats = instrumenter.stop()
    """
    
    def __init__(self, model: torch.nn.Module, sync_cuda: bool = True):
        self.model = model
        self.sync_cuda = sync_cuda
        self._original_forward = None
        self._tracker = None
        self._is_tracking = False
    
    def start(self, tracker: Optional[ComputeTracker] = None) -> ComputeTracker:
        """Start tracking compute costs."""
        if self._is_tracking:
            raise RuntimeError("Already tracking. Call stop() first.")
        
        self._tracker = tracker or ComputeTracker()
        self._original_forward = self.model.forward
        self.model.forward = create_wrapped_forward(
            self._original_forward, self._tracker, self.sync_cuda
        )
        
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._tracker.start_timer()
        
        self._is_tracking = True
        return self._tracker
    
    def stop(self) -> dict:
        """Stop tracking and return stats."""
        if not self._is_tracking:
            raise RuntimeError("Not tracking. Call start() first.")
        
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._tracker.stop_timer()
        
        self.model.forward = self._original_forward
        self._is_tracking = False
        
        return self._tracker.get_stats()
    
    @contextmanager
    def track(self, tracker: Optional[ComputeTracker] = None):
        """Context manager for tracking."""
        t = self.start(tracker)
        try:
            yield t
        finally:
            self.stop()


# Convenience function for one-off tracking
def track_compute(model: torch.nn.Module, 
                  func: Callable[[], Any],
                  tracker: Optional[ComputeTracker] = None) -> tuple:
    """
    Track compute costs for a single function call.
    
    Args:
        model: The model being used
        func: Function to execute (should use the model)
        tracker: Optional tracker (creates new one if None)
    
    Returns:
        (result, stats_dict)
    
    Example:
        result, stats = track_compute(
            model,
            lambda: model.generate(input_ids, max_new_tokens=100)
        )
    """
    if tracker is None:
        tracker = ComputeTracker()
    
    with instrument_model(model, tracker):
        result = func()
    
    return result, tracker.get_stats()
