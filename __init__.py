"""
ASMC (Annealed Sequential Monte Carlo) Experiments

This package implements ASMC sampling for LLM inference, targeting the 
distribution pi(x) = p(x)^alpha where alpha > 1.

Main components:
- asmc_sampler.py: Core ASMC implementation (sequential)
- asmc_batched.py: Batched ASMC with KV cache optimization
- asmc_math_experiment.py: Experiment script for MATH benchmark
- verifier.py: Answer verification and reranking

Key features:
1. Annealing schedule: alpha_t from 1.5 -> 4.0 over first 512 tokens
2. Defensive mixture proposal: (1-epsilon)*q_pow + epsilon*p_base
3. ESS-based systematic resampling
4. Answer-level weighted voting for output selection
5. Early stopping when mass_top >= 0.80
6. Adaptive budget: Fast pass (N=32) + Hard pass (N=96)
7. Source-weighted voting (boxed > final_line > output_block > ...)
8. Output block gate (detect unreliable output_block-only answers)
9. Verifier reranking for uncertain cases
"""

from .asmc_sampler import (
    ASMCConfig,
    ASMCSampler,
    Particle,
    compute_annealing_alpha,
    asmc_single_sample,
    build_stop_token_ids,
    weighted_voting_output,
    weighted_voting_with_verifier,
    compute_answer_masses,
    check_output_block_gate,
    SOURCE_WEIGHTS,
)

from .asmc_batched import (
    BatchedASMCSampler,
    batched_asmc_sample,
    asmc_generate_batch,
)

from .verifier import (
    verifier_rerank,
    should_trigger_verifier,
    verify_answer_batch,
)

__all__ = [
    'ASMCConfig',
    'ASMCSampler',
    'BatchedASMCSampler',
    'Particle',
    'compute_annealing_alpha',
    'asmc_single_sample',
    'batched_asmc_sample',
    'asmc_generate_batch',
    'build_stop_token_ids',
    'weighted_voting_output',
    'weighted_voting_with_verifier',
    'compute_answer_masses',
    'check_output_block_gate',
    'SOURCE_WEIGHTS',
    'verifier_rerank',
    'should_trigger_verifier',
    'verify_answer_batch',
]
