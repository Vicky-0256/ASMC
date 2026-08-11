"""Installable public namespace for cache-coherent ASMC.

The compatibility modules at the repository root remain available to the
research runner. Installed applications should import from ``asmc`` instead.
"""

from asmc_sampler import (
    ASMCConfig,
    ASMCSampler,
    Particle,
    compute_annealing_alpha,
    asmc_single_sample,
    build_stop_token_ids,
    weighted_voting_output,
    compute_answer_masses,
    check_output_block_gate,
    SOURCE_WEIGHTS,
)
from asmc_batched import (
    BatchedASMCSampler,
    batched_asmc_sample,
    asmc_generate_batch,
)
from cache import reorder_past_key_values

__all__ = [
    "ASMCConfig",
    "ASMCSampler",
    "BatchedASMCSampler",
    "Particle",
    "compute_annealing_alpha",
    "asmc_single_sample",
    "batched_asmc_sample",
    "asmc_generate_batch",
    "build_stop_token_ids",
    "weighted_voting_output",
    "compute_answer_masses",
    "check_output_block_gate",
    "reorder_past_key_values",
    "SOURCE_WEIGHTS",
]
