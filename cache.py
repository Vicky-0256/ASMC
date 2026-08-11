"""Public cache-coherence primitives.

The implementation lives next to the batched sampler for backwards
compatibility, while this module provides a stable import path for applications
that only need ancestor-based KV-cache reordering.
"""

try:  # Package import
    from .asmc_batched import reorder_past_key_values
except ImportError:  # Direct imports from the repository root
    from asmc_batched import reorder_past_key_values

__all__ = ["reorder_past_key_values"]
