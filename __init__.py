"""Annealed Sequential Monte Carlo samplers for LLM inference.

The public default targets the unconstrained power-shaped distribution.  The
historical stop-token constraints used by the paper experiments are available
only through an explicit compatibility option; see :class:`ASMCConfig`.
"""

from .asmc_sampler import (
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

from .asmc_batched import (
    BatchedASMCSampler,
    batched_asmc_sample,
    asmc_generate_batch,
)
from .cache import reorder_past_key_values

__all__ = [
    'ASMCConfig',
    'ASMCSampler',
    'BatchedASMCSampler',
    'Particle',
    'compute_annealing_alpha',
    'asmc_single_sample',
    'batched_asmc_sample',
    'asmc_generate_batch',
    'reorder_past_key_values',
    'build_stop_token_ids',
    'weighted_voting_output',
    'compute_answer_masses',
    'check_output_block_gate',
    'SOURCE_WEIGHTS',
]
