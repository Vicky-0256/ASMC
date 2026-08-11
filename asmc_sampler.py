#!/usr/bin/env python
"""
ASMC (Annealed Sequential Monte Carlo) Sampler

Based on the design document:
- Target: pi(x) ∝ p(x)^alpha* where alpha* = 4
- Annealing schedule: alpha_t from 1.5 → 4 over first 512 tokens
- Defensive mixture proposal: (1-ε)q_pow + ε*p_base
- Block-wise ESS-based systematic resampling
- Answer-level weighted voting for output
- Early stopping when mass_top >= 0.80
- Adaptive budget: Fast pass (N/2) + Hard pass (N)
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F
import numpy as np

try:  # Package import
    from .grader_utils.parse_utils import parse_answer_robust
except ImportError:  # Direct script execution from the repository root
    from grader_utils.parse_utils import parse_answer_robust


@dataclass
class ASMCConfig:
    """ASMC Sampler Configuration"""
    # Target distribution
    alpha_star: float = 4.0  # Final alpha (target = p^alpha_star)

    # Per-instance integrated-attention budget for the cache-coherent batched
    # backend. Enforcement is checked after each forward-backed generation
    # update, so final C_int may overshoot by one model forward.
    c_int_cap: Optional[float] = None
    
    # Particle settings
    n_particles: int = 64  # Fixed/hard population; adaptive fast defaults to N/2
    
    # Block settings
    block_size: int = 32  # Evaluate ESS every B tokens
    max_new_tokens: int = 3072
    
    # ESS resampling
    ess_threshold: float = 0.5  # Resample if ESS < threshold * N
    
    # Defensive mixture
    epsilon: float = 0.05  # Mixture weight for base distribution (Hard: 0.08)
    
    # Annealing schedule
    anneal_tokens: int = 512  # Anneal over first 512 tokens
    alpha_start: float = 1.5  # Starting alpha
    anneal_schedule: str = "cosine"  # "cosine" or "linear"
    
    # Output strategy
    # The paper aggregates normalized particle weights directly.  Source
    # reliability weighting is retained only as an explicit legacy mode.
    use_source_weight: bool = False
    early_stop_mass_threshold: float = 0.80  # Early stop if mass_top >= this
    early_stop_min_tokens: int = 96  # Minimum tokens before early stopping (was 64)
    # Uses the pre-resampling block ESS so a collapsed population cannot pass
    # merely because resampling reset all log weights to zero.
    early_stop_ess_frac: float = 0.25
    early_stop_min_parsed_frac: float = 0.30  # Only early-stop if n_parsed >= this * N
    early_stop_stable_checks: int = 2  # Require top_answer to be stable for this many consecutive checks
    
    # Historical EOS control. This changes the model distribution, so it is
    # opt-in in the public implementation and retained only for replaying the
    # original experiment protocol.
    legacy_stop_constraints: bool = False
    min_eos_tokens: int = 128  # Hard mask EOS before this (Fast: 128, Hard: 256)
    prefer_non_eos_until: int = 512  # Soft penalty EOS until this (Fast: 512, Hard: 768)
    eos_penalty: float = 5.0  # Logit penalty for EOS during prefer_non_eos period
    stop_token_ids: Optional[List[int]] = None  # Will be populated by build_stop_token_ids()
    
    # Rejuvenation (optional, off by default)
    enable_rejuvenation: bool = False
    rejuvenation_fraction: float = 0.25  # Fraction of particles to rejuvenate
    rejuvenation_window: int = 96  # Window size for rejuvenation
    
    # Adaptive budget. By default the fast pass uses N/2 particles and the
    # hard pass uses the configured maximum N, matching the paper protocol.
    enable_adaptive: bool = False
    fast_mass_threshold: float = 0.65  # Fast pass succeeds if mass_top >= this
    fast_n_particles: Optional[int] = None
    hard_n_particles: Optional[int] = None
    hard_anneal_tokens: int = 768
    hard_alpha_start: float = 1.3
    hard_ess_threshold: float = 0.6
    hard_min_eos_tokens: int = 256  # More conservative for hard pass
    hard_prefer_non_eos_until: int = 768
    hard_eos_penalty: float = 6.0
    # Hard pass early-stop (more conservative)
    hard_early_stop_min_tokens: int = 128
    hard_early_stop_mass_threshold: float = 0.90
    hard_early_stop_ess_frac: float = 0.30
    hard_early_stop_min_parsed_frac: float = 0.40

    def __post_init__(self) -> None:
        if not isinstance(self.use_source_weight, bool):
            raise ValueError("use_source_weight must be boolean")
        finite_fields = (
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
        for name in finite_fields:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.c_int_cap is not None:
            if not math.isfinite(float(self.c_int_cap)) or self.c_int_cap <= 0:
                raise ValueError("c_int_cap must be finite and positive when specified")
        if self.n_particles < 1:
            raise ValueError("n_particles must be positive")
        if self.fast_n_particles is None:
            self.fast_n_particles = max(1, self.n_particles // 2)
        if self.hard_n_particles is None:
            self.hard_n_particles = self.n_particles
        if not 1 <= self.fast_n_particles <= self.hard_n_particles:
            raise ValueError(
                "adaptive particle counts must satisfy "
                "1 <= fast_n_particles <= hard_n_particles"
            )
        if self.block_size < 1 or self.max_new_tokens < 1:
            raise ValueError("block_size and max_new_tokens must be positive")
        if self.anneal_tokens < 0 or self.hard_anneal_tokens < 0:
            raise ValueError("anneal token counts must be non-negative")
        if (
            self.early_stop_min_tokens < 0
            or self.hard_early_stop_min_tokens < 0
            or self.min_eos_tokens < 0
            or self.prefer_non_eos_until < 0
            or self.hard_min_eos_tokens < 0
            or self.hard_prefer_non_eos_until < 0
        ):
            raise ValueError("token thresholds must be non-negative")
        if self.early_stop_stable_checks < 1:
            raise ValueError("early_stop_stable_checks must be positive")
        if self.rejuvenation_window < 1:
            raise ValueError("rejuvenation_window must be positive")
        if self.alpha_start <= 0 or self.hard_alpha_start <= 0 or self.alpha_star <= 0:
            raise ValueError("alpha values must be positive")
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
        for name in unit_interval_fields:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be strictly between 0 and 1")
        if self.eos_penalty < 0 or self.hard_eos_penalty < 0:
            raise ValueError("EOS penalties must be non-negative")
        if self.anneal_schedule not in {"cosine", "linear"}:
            raise ValueError("anneal_schedule must be 'cosine' or 'linear'")


def build_stop_token_ids(
    tokenizer,
    include_all_special: bool = False,
) -> List[int]:
    """
    Build the token IDs that terminate a particle.

    The default matches ordinary autoregressive generation and includes only
    EOS. ``include_all_special=True`` reproduces the historical ASMC behavior
    that also treated padding and every special token as a stop token.
    """
    ids = set()
    
    # Add EOS token (must have)
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_ids is not None:
        if isinstance(eos_ids, (list, tuple, set)):
            ids.update(int(token_id) for token_id in eos_ids)
        else:
            ids.add(int(eos_ids))
    
    # Add pad token if different from EOS
    if include_all_special and getattr(tokenizer, "pad_token_id", None) is not None:
        ids.add(int(tokenizer.pad_token_id))
    
    # Add all special tokens as insurance
    # (prevents early termination from any special token)
    if include_all_special:
        try:
            for tid in tokenizer.all_special_ids:
                ids.add(int(tid))
        except Exception:
            pass
    
    return sorted(ids)


def compute_annealing_alpha(
    t: int,
    alpha_start: float,
    alpha_star: float,
    anneal_tokens: int,
    schedule: str = "cosine"
) -> float:
    """
    Compute annealing coefficient alpha_t at token position t.
    
    For t < anneal_tokens: interpolate from alpha_start to alpha_star
    For t >= anneal_tokens: return alpha_star
    
    Args:
        t: Current token position (0-indexed from start of generation)
        alpha_start: Starting alpha value
        alpha_star: Final alpha value
        anneal_tokens: Number of tokens over which to anneal
        schedule: "cosine" or "linear"
    
    Returns:
        alpha_t: Annealing coefficient at position t
    """
    if t >= anneal_tokens:
        return alpha_star
    
    progress = t / anneal_tokens  # 0 to 1
    
    if schedule == "cosine":
        # alpha_t = alpha_star - (alpha_star - alpha_start) * (1 + cos(pi * progress)) / 2
        # This starts at alpha_start and ends at alpha_star
        factor = (1 + math.cos(math.pi * progress)) / 2
        return alpha_star - (alpha_star - alpha_start) * factor
    elif schedule == "linear":
        return alpha_start + (alpha_star - alpha_start) * progress
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def compute_power_proposal_logprobs(
    logits: torch.Tensor,
    alpha_t: float,
    epsilon: float
) -> torch.Tensor:
    """
    Compute log probabilities for the defensive mixture power proposal.
    
    q_t(v|h) = (1-ε) * q_pow_t(v|h) + ε * p(v|h)
    
    where q_pow_t(v|h) = softmax(alpha_t * log p(v|h))
    
    Args:
        logits: Raw logits from model (vocab_size,)
        alpha_t: Current annealing coefficient
        epsilon: Defensive mixture weight
    
    Returns:
        log_q: Log probabilities under the mixture proposal (vocab_size,)
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between 0 and 1")

    # Base log probabilities
    log_p = F.log_softmax(logits, dim=-1)
    
    # Power proposal: softmax(alpha_t * log_p) = softmax(alpha_t * (logits - log_Z))
    # Since softmax is shift-invariant: softmax(alpha_t * logits)
    scaled_logits = alpha_t * logits
    log_q_pow = F.log_softmax(scaled_logits, dim=-1)
    
    # Mixture: (1-ε) * q_pow + ε * p
    # In log space: log((1-ε) * exp(log_q_pow) + ε * exp(log_p))
    # = logsumexp([log(1-ε) + log_q_pow, log(ε) + log_p])
    
    log_one_minus_eps = math.log(1 - epsilon)
    log_eps = math.log(epsilon)
    
    # Stack for logsumexp: (2, vocab_size)
    stacked = torch.stack([
        log_one_minus_eps + log_q_pow,
        log_eps + log_p
    ], dim=0)
    
    log_q = torch.logsumexp(stacked, dim=0)
    
    return log_q


def sample_from_power_proposal(
    logits: torch.Tensor,
    alpha_t: float,
    epsilon: float
) -> Tuple[int, float, float]:
    """
    Sample a token from the defensive mixture power proposal.
    
    Returns:
        tok: Sampled token id
        log_q_tok: log q(tok) under the mixture proposal
        log_p_tok: log p(tok) under the base model
    """
    log_q = compute_power_proposal_logprobs(logits, alpha_t, epsilon)
    log_p = F.log_softmax(logits, dim=-1)
    
    # Sample from q
    probs_q = log_q.exp()
    tok = int(torch.multinomial(probs_q, 1).item())
    
    log_q_tok = float(log_q[tok].item())
    log_p_tok = float(log_p[tok].item())
    
    return tok, log_q_tok, log_p_tok


def systematic_resample(weights: List[float], n: int) -> List[int]:
    """
    Systematic resampling (low variance resampling).
    
    Args:
        weights: Normalized weights (sum to 1)
        n: Number of samples to draw
    
    Returns:
        indices: List of resampled indices
    """
    # Use a single random offset for all positions (systematic)
    u0 = random.random() / n
    positions = [u0 + i / n for i in range(n)]
    
    indices = []
    cumsum = 0.0
    i = 0
    m = len(weights)
    
    for pos in positions:
        # Move forward until we find the right bin
        while i < m - 1 and cumsum + weights[i] < pos:
            cumsum += weights[i]
            i += 1
        indices.append(i)
    
    return indices


def compute_ess_from_logw(logw: List[float]) -> float:
    """
    Compute Effective Sample Size from log weights.
    
    ESS = 1 / sum(w_i^2) where w_i are normalized weights
    """
    if not logw:
        return 0.0
    
    # Normalize in log space
    max_logw = max(logw)
    weights = [math.exp(lw - max_logw) for lw in logw]
    total = sum(weights)
    
    if total == 0:
        return 0.0
    
    weights = [w / total for w in weights]
    sum_sq = sum(w * w for w in weights)
    
    return 1.0 / (sum_sq + 1e-12)


def normalize_logweights(logw: List[float]) -> List[float]:
    """Normalize log weights to get proper weights summing to 1."""
    if not logw:
        return []
    max_logw = max(logw)
    weights = [math.exp(lw - max_logw) for lw in logw]
    total = sum(weights)
    return [w / total for w in weights]


@dataclass
class Particle:
    """A single particle in SMC."""
    tokens: List[int]  # Full sequence including context
    log_weight: float  # Current (unnormalized) log weight
    log_p_sum: float  # Sum of log p(x_t) under base model
    log_q_sum: float  # Sum of log q(x_t) under proposal
    finished: bool  # Whether this particle has generated EOS
    
    def copy(self) -> 'Particle':
        return Particle(
            tokens=self.tokens.copy(),
            log_weight=self.log_weight,
            log_p_sum=self.log_p_sum,
            log_q_sum=self.log_q_sum,
            finished=self.finished
        )


@torch.no_grad()
def asmc_sample(
    model,
    tokenizer,
    context: List[int],
    config: ASMCConfig,
    device,
    verbose: bool = False,
    tracker=None,
) -> Tuple[List[Particle], Dict[str, Any]]:
    """
    Run ASMC sampling.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        context: Input token ids (prompt)
        config: ASMC configuration
        device: Compute device
        verbose: Print progress
        tracker: Optional ComputeTracker used by model instrumentation. The
            sequential reference backend does not support C_int caps because
            stopping within a population step would return partially updated
            particles; use the batched public backend for capped runs.
    
    Returns:
        particles: Final list of particles
        diagnostics: Dictionary with diagnostics info
    """
    if config.c_int_cap is not None:
        raise ValueError(
            "c_int_cap is supported only by the cache-coherent batched backend"
        )

    N = config.n_particles
    c = len(context)
    max_len = c + config.max_new_tokens
    
    # Build stop token IDs if not already set
    if config.stop_token_ids is None:
        config.stop_token_ids = build_stop_token_ids(
            tokenizer,
            include_all_special=config.legacy_stop_constraints,
        )
    
    # Initialize particles
    particles = [
        Particle(
            tokens=context.copy(),
            log_weight=0.0,
            log_p_sum=0.0,
            log_q_sum=0.0,
            finished=False
        )
        for _ in range(N)
    ]
    
    # Diagnostics
    diagnostics = {
        "n_particles": N,
        "n_resamples": 0,
        "ess_history": [],
        "alpha_history": [],
        "tokens_generated": 0,
        "early_stopped": False,
        "early_stop_token": None,
        "n_eos_sampled_blocks": [],  # Track EOS sampling per block
        "n_finished_blocks": [],  # Track finished particles per block
        "n_active_blocks": [],  # Track active particles per block
        "n_parsed_blocks": [],  # Track parsed answers per block
        "ess_blocks": [],  # ESS at each block boundary
        "stop_reason": None,  # early_stop / max_len / all_finished / error
        "c_int_cap": config.c_int_cap,
        "budget_exhausted": False,
        "budget_exhausted_at_token": None,
        "gen_len_best": 0,  # Length of best particle completion
        "non_special_len_best": 0,  # Non-special token length
    }

    if config.c_int_cap is not None and tracker.C_int >= config.c_int_cap:
        diagnostics["budget_exhausted"] = True
        diagnostics["budget_exhausted_at_token"] = -1
        diagnostics["stop_reason"] = "budget_exhausted"
        return particles, diagnostics
    
    # Token-by-token generation
    current_block_start = 0
    n_eos_this_block = 0  # Track EOS sampling within current block
    active_indices = list(range(N))  # Initialize: all particles active
    
    for t_gen in range(config.max_new_tokens):
        t = t_gen  # Token position from start of generation
        current_pos = c + t  # Absolute position in sequence
        
        # Compute annealing alpha
        alpha_t = compute_annealing_alpha(
            t, config.alpha_start, config.alpha_star,
            config.anneal_tokens, config.anneal_schedule
        )
        diagnostics["alpha_history"].append(alpha_t)
        
        # Count active particles
        active_indices = [i for i, p in enumerate(particles) if not p.finished]
        
        if not active_indices:
            break
        
        # Sample next token for each active particle
        n_eos_this_step = 0  # Track EOS sampling for diagnostics
        
        for i in active_indices:
            p = particles[i]
            
            # Get logits
            input_ids = torch.tensor([p.tokens], dtype=torch.long, device=device)
            if input_ids.size(1) > model.config.max_position_embeddings:
                input_ids = input_ids[:, -model.config.max_position_embeddings:]
            
            output = model(input_ids)
            logits = output.logits[0, -1, :].clone()  # (vocab_size,) - clone for safe modification
            
            # 🔴 CRITICAL FIX: Mask/penalize stop tokens to prevent Problem 94
            # This prevents 35% of particles from hitting EOS at t=32
            if config.legacy_stop_constraints and config.stop_token_ids:
                if t_gen < config.min_eos_tokens:
                    # Hard mask: completely prevent EOS generation
                    for stop_id in config.stop_token_ids:
                        logits[stop_id] = -1e30
                elif t_gen < config.prefer_non_eos_until:
                    # Soft penalty: discourage but allow EOS
                    for stop_id in config.stop_token_ids:
                        logits[stop_id] = logits[stop_id] - config.eos_penalty
            
            # Sample from power proposal
            tok, log_q_tok, log_p_tok = sample_from_power_proposal(
                logits, alpha_t, config.epsilon
            )
            
            # Update particle
            p.tokens.append(tok)
            p.log_q_sum += log_q_tok
            p.log_p_sum += log_p_tok
            
            # Incremental weight update: Δlog w = α* log p - log q
            delta_logw = config.alpha_star * log_p_tok - log_q_tok
            p.log_weight += delta_logw
            
            # Check for EOS (check all stop tokens)
            if config.stop_token_ids and tok in config.stop_token_ids:
                p.finished = True
                n_eos_this_step += 1


        # Accumulate EOS count for this block
        n_eos_this_block += n_eos_this_step
        diagnostics["tokens_generated"] = t_gen + 1

        # Block-wise ESS evaluation and resampling
        if (t_gen + 1) % config.block_size == 0 or t_gen == config.max_new_tokens - 1:
            # Compute ESS
            logw = [p.log_weight for p in particles]
            ess = compute_ess_from_logw(logw)
            ess_for_early_stop = ess
            diagnostics["ess_history"].append(ess)
            diagnostics["ess_blocks"].append(ess)
            
            # Track diagnostics
            n_finished = sum(1 for p in particles if p.finished)
            n_active = len(active_indices)
            diagnostics["n_eos_sampled_blocks"].append(n_eos_this_block)
            diagnostics["n_finished_blocks"].append(n_finished)
            diagnostics["n_active_blocks"].append(n_active)
            
            # Track parsed answers at this block
            try:
                answer_masses, n_parsed, source_counts = compute_answer_masses(
                    particles,
                    tokenizer,
                    c,
                    use_source_weight=config.use_source_weight,
                )
            except Exception:
                n_parsed = 0
                answer_masses = {}
                source_counts = {}
            diagnostics["n_parsed_blocks"].append(n_parsed)
            
            # Reset block EOS counter
            n_eos_this_block = 0
            
            if verbose:
                print(f"  Block {(t_gen+1)//config.block_size}: t={t_gen+1}, alpha={alpha_t:.2f}, ESS={ess:.2f}, active={n_active}, n_parsed={n_parsed}")
            
            # ====== STEP 1: RESAMPLE FIRST (if ESS too low) ======
            # This must happen BEFORE early-stop check to ensure healthy particle set
            if ess < config.ess_threshold * N:
                if verbose:
                    print(f"  Resampling (ESS={ess:.2f} < {config.ess_threshold * N:.2f})")
                
                # Normalize weights
                weights = normalize_logweights(logw)
                
                # Systematic resampling
                indices = systematic_resample(weights, N)
                
                # Create new particles
                new_particles = [particles[idx].copy() for idx in indices]
                
                # Reset log weights after resampling
                for p in new_particles:
                    p.log_weight = 0.0
                
                particles = new_particles
                diagnostics["n_resamples"] += 1
                
                # Recompute ESS after resampling (should be ~N now)
                logw = [p.log_weight for p in particles]
                ess = compute_ess_from_logw(logw)
                
                # Recompute answer masses after resampling
                try:
                    answer_masses, n_parsed, source_counts = compute_answer_masses(
                        particles,
                        tokenizer,
                        c,
                        use_source_weight=config.use_source_weight,
                    )
                except Exception:
                    answer_masses = {}
                    n_parsed = 0
                    source_counts = {}
                
                # Optional: Rejuvenation
                if config.enable_rejuvenation:
                    n_rejuv = int(N * config.rejuvenation_fraction)
                    rejuv_indices = random.sample(range(N), n_rejuv)
                    
                    for idx in rejuv_indices:
                        particles[idx] = rejuvenate_particle(
                            model, tokenizer, particles[idx],
                            config, c, device, tracker=tracker
                        )
                        if (
                            config.c_int_cap is not None
                            and tracker.C_int >= config.c_int_cap
                        ):
                            diagnostics["budget_exhausted"] = True
                            diagnostics["budget_exhausted_at_token"] = t_gen
                            diagnostics["stop_reason"] = "budget_exhausted"
                            break

                    if diagnostics["budget_exhausted"]:
                        break
            
            # ====== STEP 2: EARLY-STOP CHECK (only if ESS is healthy) ======
            # CRITICAL: Only allow early-stop if ESS >= threshold (prevents single-particle domination)
            if t_gen + 1 >= config.early_stop_min_tokens:
                ess_threshold_for_stop = config.early_stop_ess_frac * N
                parsed_threshold_for_stop = config.early_stop_min_parsed_frac * N
                
                # Gate 1: ESS must be healthy
                if ess_for_early_stop < ess_threshold_for_stop:
                    if verbose:
                        print(
                            "  Early-stop blocked: pre-resampling "
                            f"ESS={ess_for_early_stop:.2f} < "
                            f"{ess_threshold_for_stop:.1f}"
                        )
                # Gate 2: Enough particles must have parsed answers
                elif n_parsed < parsed_threshold_for_stop:
                    if verbose:
                        print(f"  Early-stop blocked: n_parsed={n_parsed} < {parsed_threshold_for_stop:.1f}")
                # Gate 3: Answer mass threshold
                elif answer_masses:
                    top_answer, top_mass = max(answer_masses.items(), key=lambda x: x[1])
                    
                    # Gate 4: Stability check - top answer must be stable
                    prev_top = diagnostics.get("_prev_top_answer", None)
                    stable_count = diagnostics.get("_stable_count", 0)
                    
                    if top_answer == prev_top:
                        stable_count += 1
                    else:
                        stable_count = 1
                    
                    diagnostics["_prev_top_answer"] = top_answer
                    diagnostics["_stable_count"] = stable_count
                    
                    if top_mass >= config.early_stop_mass_threshold:
                        if stable_count >= config.early_stop_stable_checks:
                            diagnostics["early_stopped"] = True
                            diagnostics["early_stop_token"] = t_gen
                            diagnostics["stop_reason"] = "early_stop"
                            if verbose:
                                print(
                                    f"  Early stop at t={t_gen}: "
                                    f"mass={top_mass:.2f}, "
                                    f"pre-resampling ESS={ess_for_early_stop:.2f}, "
                                    f"stable={stable_count}"
                                )
                            break
                        else:
                            if verbose:
                                print(f"  Early-stop pending: mass={top_mass:.2f} but stable_count={stable_count} < {config.early_stop_stable_checks}")
    
    # Set stop_reason if not already set
    if diagnostics["stop_reason"] is None:
        if len(active_indices) == 0:
            diagnostics["stop_reason"] = "all_finished"
        elif diagnostics["tokens_generated"] >= config.max_new_tokens:
            diagnostics["stop_reason"] = "max_len"
        else:
            diagnostics["stop_reason"] = "unknown"
    
    return particles, diagnostics


@torch.no_grad()
def rejuvenate_particle(
    model,
    tokenizer,
    particle: Particle,
    config: ASMCConfig,
    context_len: int,
    device,
    tracker=None,
) -> Particle:
    """
    Rejuvenate a particle by resampling its tail.
    
    This performs a light MH move on the last L tokens.
    For simplicity, we always accept the new proposal (Gibbs-style).
    """
    L = config.rejuvenation_window
    gen_len = len(particle.tokens) - context_len
    
    if gen_len <= L:
        # Not enough tokens to rejuvenate, skip
        return particle
    
    # Keep tokens up to the rejuvenation point
    keep_len = len(particle.tokens) - L
    new_tokens = particle.tokens[:keep_len]
    
    # We need to recompute log_p and log_q for the kept portion
    # For efficiency, we estimate: (kept_len / gen_len) * original_sum
    kept_gen_len = keep_len - context_len
    ratio = kept_gen_len / gen_len if gen_len > 0 else 0.0
    kept_log_p_sum = particle.log_p_sum * ratio
    kept_log_q_sum = particle.log_q_sum * ratio
    
    # Re-generate the tail
    new_log_p_sum = 0.0
    new_log_q_sum = 0.0
    
    for _ in range(L):
        if len(new_tokens) >= context_len + config.max_new_tokens:
            break
        
        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=device)
        if input_ids.size(1) > model.config.max_position_embeddings:
            input_ids = input_ids[:, -model.config.max_position_embeddings:]
        
        output = model(input_ids)
        logits = output.logits[0, -1, :]
        
        # Use full alpha for rejuvenation
        tok, log_q_tok, log_p_tok = sample_from_power_proposal(
            logits, config.alpha_star, config.epsilon
        )
        
        new_tokens.append(tok)
        new_log_p_sum += log_p_tok
        new_log_q_sum += log_q_tok

        if (
            config.c_int_cap is not None
            and tracker is not None
            and tracker.C_int >= config.c_int_cap
        ):
            break
        
        if tok == tokenizer.eos_token_id:
            break
    
    # Create new particle with updated sums
    # Note: Weight is kept from resampling (already reset to 0)
    return Particle(
        tokens=new_tokens,
        log_weight=particle.log_weight,
        log_p_sum=kept_log_p_sum + new_log_p_sum,
        log_q_sum=kept_log_q_sum + new_log_q_sum,
        finished=new_tokens[-1] == tokenizer.eos_token_id if new_tokens else False
    )


# Source weights for answer reliability scoring
SOURCE_WEIGHTS = {
    "boxed": 1.0,
    "final_line": 0.95,
    "output_block": 0.80,  # Reduced - needs independent verification
    "print_output": 0.75,
    "inline_math": 0.65,
    "is_pattern": 0.55,
    "fallback_number": 0.10,
}


def compute_answer_masses(
    particles: List[Particle],
    tokenizer,
    context_len: int,
    use_source_weight: bool = False,
) -> Tuple[Dict[str, float], int, Dict[str, Dict[str, int]]]:
    """
    Compute answer-level weighted masses with source tracking.
    
    Parse answers from each particle and aggregate weights by answer.
    Optionally weight by source reliability.
    
    Returns:
        answer_masses: Dict mapping answer -> weighted mass
        n_parsed: Number of particles with successfully parsed answers
        source_counts: Dict mapping answer -> {source: count}
    """
    # Get normalized weights
    logw = [p.log_weight for p in particles]
    weights = normalize_logweights(logw)
    
    answer_masses = {}
    source_counts = {}  # Track source distribution per answer
    n_parsed = 0
    
    for i, particle in enumerate(particles):
        # Decode completion
        completion_ids = particle.tokens[context_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        
        # Parse answer with source info
        answer, source = parse_answer_robust(completion, return_source=True)
        
        if answer is not None:
            n_parsed += 1
            answer_key = str(answer).strip()
            
            # Source-weighted contribution
            src_weight = SOURCE_WEIGHTS.get(source, 0.5) if use_source_weight else 1.0
            contribution = weights[i] * src_weight
            
            answer_masses[answer_key] = answer_masses.get(answer_key, 0) + contribution
            
            # Track source distribution
            if answer_key not in source_counts:
                source_counts[answer_key] = {}
            source_counts[answer_key][source] = source_counts[answer_key].get(source, 0) + 1
    
    return answer_masses, n_parsed, source_counts


def check_output_block_gate(
    answer_masses: Dict[str, float],
    source_counts: Dict[str, Dict[str, int]],
    n_particles: int,
    threshold: float = 0.3,
) -> Tuple[bool, str]:
    """
    Check if top answer is overly reliant on output_block source.
    
    Gate conditions (any passes):
    1. Has boxed or final_line support -> PASS
    2. Non-output_block sources >= threshold * n_particles -> PASS
    3. Only output_block with < 10% other sources -> FAIL
    
    Returns:
        (is_reliable, reason)
    """
    if not answer_masses:
        return True, "no_answers"
    
    top_answer = max(answer_masses, key=answer_masses.get)
    sources = source_counts.get(top_answer, {})
    
    total_for_answer = sum(sources.values())
    output_block_count = sources.get("output_block", 0)
    boxed_count = sources.get("boxed", 0)
    final_line_count = sources.get("final_line", 0)
    non_output_count = total_for_answer - output_block_count
    
    # Gate 1: Has reliable source (boxed or final_line)
    if boxed_count > 0 or final_line_count > 0:
        return True, "has_reliable_source"
    
    # Gate 2: Enough diverse sources
    if non_output_count >= threshold * n_particles:
        return True, "enough_diverse_sources"
    
    # Gate 3: Fail if only output_block dominates
    if output_block_count > 0 and non_output_count < 0.1 * n_particles:
        return False, "output_block_only"
    
    return True, "default_pass"


def weighted_voting_output(
    particles: List[Particle],
    tokenizer,
    context_len: int,
    alpha_star: float = 4.0,
    use_source_weight: bool = False,
) -> Tuple[str, Particle, Dict[str, Any]]:
    """
    Select output using answer-level weighted voting.

    By default, normalized particle weights are aggregated directly, matching
    the paper. ``use_source_weight=True`` enables the historical parser-source
    reliability multipliers.
    
    Returns:
        best_answer: The answer with highest weighted mass
        best_particle: The particle with highest log pi among those with the best answer
        vote_info: Dictionary with voting information
    """
    N = len(particles)
    
    # Get normalized weights
    logw = [p.log_weight for p in particles]
    weights = normalize_logweights(logw)
    
    # Group particles by answer with source tracking
    answer_to_particles = {}  # answer -> list of (particle_idx, weight, log_pi, source)
    source_counts = {}  # answer -> {source: count}
    n_parsed = 0
    
    for i, particle in enumerate(particles):
        completion_ids = particle.tokens[context_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        answer, source = parse_answer_robust(completion, return_source=True)
        
        if answer is not None:
            n_parsed += 1
            answer_key = str(answer).strip()
            log_pi = particle.log_p_sum * alpha_star
            
            if answer_key not in answer_to_particles:
                answer_to_particles[answer_key] = []
                source_counts[answer_key] = {}
            
            answer_to_particles[answer_key].append((i, weights[i], log_pi, source))
            source_counts[answer_key][source] = source_counts[answer_key].get(source, 0) + 1
    
    if not answer_to_particles:
        # No valid answers parsed, return highest weight particle
        best_idx = max(range(N), key=lambda i: logw[i])
        return None, particles[best_idx], {
            "no_valid_answers": True,
            "n_answers": 0,
            "n_parsed": 0,
            "gate_passed": True,
            "gate_reason": "no_answers",
            "vote_method": "weighted",
            "use_source_weight": use_source_weight,
        }
    
    # Aggregate normalized particle mass, with optional legacy source weights.
    answer_masses = {}
    for ans, items in answer_to_particles.items():
        total_mass = 0.0
        for _, weight, _, source in items:
            src_weight = SOURCE_WEIGHTS.get(source, 0.5) if use_source_weight else 1.0
            total_mass += weight * src_weight
        answer_masses[ans] = total_mass
    
    # Check output_block gate
    gate_passed, gate_reason = check_output_block_gate(
        answer_masses, source_counts, N
    )
    
    # Find best answer
    best_answer = max(answer_masses, key=answer_masses.get)
    best_mass = answer_masses[best_answer]
    
    # Among particles with best answer, pick one with highest log_pi
    candidates = answer_to_particles[best_answer]
    best_idx = max(candidates, key=lambda x: x[2])[0]
    best_particle = particles[best_idx]
    
    vote_info = {
        "answer_masses": answer_masses,
        "best_answer": best_answer,
        "best_mass": best_mass,
        "n_unique_answers": len(answer_masses),
        "n_answers": len(answer_masses),
        "n_parsed": n_parsed,
        "best_particle_idx": best_idx,
        "source_counts": source_counts,
        "gate_passed": gate_passed,
        "gate_reason": gate_reason,
        "vote_method": "weighted",
        "use_source_weight": use_source_weight,
    }
    
    return best_answer, best_particle, vote_info


def unweighted_majority_voting(
    particles: List[Particle],
    tokenizer,
    context_len: int,
    alpha_star: float = 4.0,
    use_source_weight: bool = False,  # Kept for API consistency; pure counts ignore it.
) -> Tuple[str, Particle, Dict[str, Any]]:
    """
    Unweighted majority voting: each particle = 1 vote (ignore importance weights).
    Tie-break by sum of normalized importance weights.

    This is an ablation to test if importance weighting in aggregation hurts accuracy.

    Args:
        particles: List of ASMC particles
        tokenizer: Tokenizer for decoding
        context_len: Length of context (prompt)
        alpha_star: Temperature for log_pi calculation (used for particle selection)
        use_source_weight: Unused in majority voting, kept for API consistency

    Returns:
        best_answer: The majority vote answer
        best_particle: The particle with highest log_pi among those with best answer
        vote_info: Dictionary with voting statistics
    """
    from collections import Counter

    N = len(particles)
    logw = [p.log_weight for p in particles]
    weights = normalize_logweights(logw)  # For tie-break only

    # Count answers (each particle = 1 vote)
    answer_counts = Counter()
    answer_to_particles = {}  # answer -> [(idx, weight, log_pi, source)]
    n_parsed = 0

    for i, particle in enumerate(particles):
        completion_ids = particle.tokens[context_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        answer, source = parse_answer_robust(completion, return_source=True)

        if answer is not None:
            n_parsed += 1
            answer_key = str(answer).strip()
            answer_counts[answer_key] += 1
            log_pi = particle.log_p_sum * alpha_star

            if answer_key not in answer_to_particles:
                answer_to_particles[answer_key] = []
            answer_to_particles[answer_key].append((i, weights[i], log_pi, source))

    # Handle empty case: no valid answers parsed
    if not answer_counts:
        best_idx = max(range(N), key=lambda i: logw[i])
        return None, particles[best_idx], {
            "no_valid_answers": True,
            "n_answers": 0,
            "n_parsed": 0,
            "vote_method": "majority",
        }

    # Find max count
    max_count = answer_counts.most_common(1)[0][1]
    tied_answers = [ans for ans, cnt in answer_counts.items() if cnt == max_count]

    # Tie-break by sum of normalized importance weights
    if len(tied_answers) > 1:
        best_answer = max(tied_answers,
            key=lambda a: sum(w for _, w, _, _ in answer_to_particles[a]))
    else:
        best_answer = tied_answers[0]

    best_count = answer_counts[best_answer]

    # Select best particle (highest log_pi among those with best answer)
    candidates = answer_to_particles[best_answer]
    best_idx = max(candidates, key=lambda x: x[2])[0]
    best_particle = particles[best_idx]

    vote_info = {
        "answer_counts": dict(answer_counts),
        "best_answer": best_answer,
        "best_count": best_count,
        "best_mass": best_count / N,  # For consistency with weighted voting
        "n_unique_answers": len(answer_counts),
        "n_answers": len(answer_counts),
        "n_parsed": n_parsed,
        "n_tied": len(tied_answers),
        "vote_method": "majority",
        "best_particle_idx": best_idx,
    }

    return best_answer, best_particle, vote_info


class ASMCSampler:
    """
    ASMC Sampler with adaptive budget.
    
    Provides two-stage sampling. By default the fast pass uses N/2 particles
    and the hard pass uses N; optional rejuvenation follows the parent config.
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def sample(
        self,
        context: List[int],
        config: Optional[ASMCConfig] = None,
        verbose: bool = False,
        tracker=None,
    ) -> Tuple[List[Particle], str, Particle, Dict[str, Any]]:
        """
        Run ASMC sampling with optional adaptive budget.
        
        Returns:
            particles: Final list of particles
            best_answer: Best answer from weighted voting
            best_particle: Best particle
            full_diagnostics: Complete diagnostics
        """
        if config is None:
            config = ASMCConfig()
        
        # Build stop_token_ids if not already set
        if config.stop_token_ids is None:
            config.stop_token_ids = build_stop_token_ids(
                self.tokenizer,
                include_all_special=config.legacy_stop_constraints,
            )
        
        c = len(context)
        
        if not config.enable_adaptive:
            # Single pass
            particles, diagnostics = asmc_sample(
                self.model,
                self.tokenizer,
                context,
                config,
                self.device,
                verbose,
                tracker=tracker,
            )
            best_answer, best_particle, vote_info = weighted_voting_output(
                particles,
                self.tokenizer,
                c,
                config.alpha_star,
                use_source_weight=config.use_source_weight,
            )
            diagnostics["vote_info"] = vote_info
            return particles, best_answer, best_particle, diagnostics
        
        # Adaptive: Fast pass first
        fast_config = ASMCConfig(
            c_int_cap=config.c_int_cap,
            n_particles=config.fast_n_particles,
            alpha_star=config.alpha_star,
            block_size=config.block_size,
            max_new_tokens=config.max_new_tokens,
            ess_threshold=config.ess_threshold,
            epsilon=config.epsilon,
            anneal_tokens=config.anneal_tokens,
            alpha_start=config.alpha_start,
            anneal_schedule=config.anneal_schedule,
            use_source_weight=config.use_source_weight,
            # Early stop settings (with new gates)
            early_stop_mass_threshold=config.early_stop_mass_threshold,
            early_stop_min_tokens=config.early_stop_min_tokens,
            early_stop_ess_frac=config.early_stop_ess_frac,
            early_stop_min_parsed_frac=config.early_stop_min_parsed_frac,
            early_stop_stable_checks=config.early_stop_stable_checks,
            # Historical EOS control
            legacy_stop_constraints=config.legacy_stop_constraints,
            min_eos_tokens=config.min_eos_tokens,
            prefer_non_eos_until=config.prefer_non_eos_until,
            eos_penalty=config.eos_penalty,
            stop_token_ids=config.stop_token_ids,  # Share stop tokens
            enable_rejuvenation=False,  # No rejuv in fast pass
            enable_adaptive=False,
        )
        
        if verbose:
            print(f"=== Fast Pass (N={config.fast_n_particles}) ===")
        
        fast_particles, fast_diag = asmc_sample(
            self.model,
            self.tokenizer,
            context,
            fast_config,
            self.device,
            verbose,
            tracker=tracker,
        )
        
        best_answer, best_particle, vote_info = weighted_voting_output(
            fast_particles,
            self.tokenizer,
            c,
            config.alpha_star,
            use_source_weight=config.use_source_weight,
        )
        fast_diag["vote_info"] = vote_info

        if fast_diag.get("budget_exhausted", False):
            # Keep pass_type orthogonal to budget status for audit compatibility.
            fast_diag["pass_type"] = "fast"
            return fast_particles, best_answer, best_particle, fast_diag
        
        # Check if fast pass succeeded
        mass_top = vote_info.get("best_mass", 0.0)
        
        if mass_top >= config.fast_mass_threshold:
            if verbose:
                print(f"Fast pass succeeded: mass_top={mass_top:.2f} >= {config.fast_mass_threshold}")
            fast_diag["pass_type"] = "fast"
            return fast_particles, best_answer, best_particle, fast_diag
        
        # Hard pass needed
        if verbose:
            print(f"Fast pass insufficient: mass_top={mass_top:.2f} < {config.fast_mass_threshold}")
            print(f"=== Hard Pass (N={config.hard_n_particles}) ===")
        
        hard_config = ASMCConfig(
            c_int_cap=config.c_int_cap,
            n_particles=config.hard_n_particles,
            alpha_star=config.alpha_star,
            block_size=config.block_size,
            max_new_tokens=config.max_new_tokens,
            ess_threshold=config.hard_ess_threshold,
            epsilon=0.08,  # Higher epsilon for hard pass
            anneal_tokens=config.hard_anneal_tokens,
            alpha_start=config.hard_alpha_start,
            anneal_schedule=config.anneal_schedule,
            use_source_weight=config.use_source_weight,
            # Early stop settings (more conservative for hard pass)
            early_stop_mass_threshold=config.hard_early_stop_mass_threshold,
            early_stop_min_tokens=config.hard_early_stop_min_tokens,
            early_stop_ess_frac=config.hard_early_stop_ess_frac,
            early_stop_min_parsed_frac=config.hard_early_stop_min_parsed_frac,
            early_stop_stable_checks=config.early_stop_stable_checks,
            # Historical EOS control (more conservative for hard pass)
            legacy_stop_constraints=config.legacy_stop_constraints,
            min_eos_tokens=config.hard_min_eos_tokens,
            prefer_non_eos_until=config.hard_prefer_non_eos_until,
            eos_penalty=config.hard_eos_penalty,
            stop_token_ids=config.stop_token_ids,  # Share stop tokens
            enable_rejuvenation=config.enable_rejuvenation,
            rejuvenation_fraction=config.rejuvenation_fraction,
            rejuvenation_window=config.rejuvenation_window,
            enable_adaptive=False,
        )
        
        hard_particles, hard_diag = asmc_sample(
            self.model,
            self.tokenizer,
            context,
            hard_config,
            self.device,
            verbose,
            tracker=tracker,
        )
        
        best_answer, best_particle, vote_info = weighted_voting_output(
            hard_particles,
            self.tokenizer,
            c,
            config.alpha_star,
            use_source_weight=config.use_source_weight,
        )
        hard_diag["vote_info"] = vote_info
        hard_diag["fast_diag"] = fast_diag  # Include fast pass diagnostics
        hard_diag["pass_type"] = "hard"
        
        return hard_particles, best_answer, best_particle, hard_diag


# Convenience function for single-sample interface
@torch.no_grad()
def asmc_single_sample(
    model,
    tokenizer,
    context: List[int],
    device,
    config: Optional[ASMCConfig] = None,
    verbose: bool = False,
    tracker=None,
) -> Tuple[List[int], str, Dict[str, Any]]:
    """
    Run ASMC and return a single best sequence.
    
    Returns:
        best_tokens: Token ids of best sequence
        best_answer: Parsed answer
        diagnostics: Full diagnostics
    """
    sampler = ASMCSampler(model, tokenizer, device)
    
    if config is None:
        config = ASMCConfig()
    
    particles, best_answer, best_particle, diagnostics = sampler.sample(
        context, config, verbose, tracker=tracker
    )
    
    return best_particle.tokens, best_answer, diagnostics
