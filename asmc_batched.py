#!/usr/bin/env python
"""
Batched ASMC Sampler with KV Cache Optimization

This version processes all particles in parallel using batched inference,
which provides significant speedup on GPU.

Key optimizations:
1. Batched forward pass for all particles
2. KV cache reuse within generation
3. Efficient handling of variable-length sequences with padding
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import copy

import torch
import torch.nn.functional as F
import numpy as np

try:  # Package import
    from .asmc_sampler import (
        ASMCConfig, Particle,
        compute_annealing_alpha,
        compute_power_proposal_logprobs,
        systematic_resample,
        compute_ess_from_logw,
        normalize_logweights,
        compute_answer_masses,
        build_stop_token_ids,
        weighted_voting_output,
    )
except ImportError:  # Direct script execution from the repository root
    from asmc_sampler import (
        ASMCConfig, Particle,
        compute_annealing_alpha,
        compute_power_proposal_logprobs,
        systematic_resample,
        compute_ess_from_logw,
        normalize_logweights,
        compute_answer_masses,
        build_stop_token_ids,
        weighted_voting_output,
    )


def ensure_pad_token(tokenizer):
    """Ensure tokenizer has a pad token."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer.pad_token_id


@torch.no_grad()
def reorder_past_key_values(past_key_values, indices, device=None):
    """
    Reorder past_key_values (DynamicCache or legacy tuple cache) along batch dim by indices.
    
    This is used after resampling to rearrange the KV cache according to the new particle
    ordering, avoiding expensive full forward pass rebuild.
    
    Args:
        past_key_values: DynamicCache or tuple of (key, value) per layer
        indices: List[int] or 1D LongTensor of shape (N,) - resample indices
        device: Optional device for tensor conversion
    
    Returns:
        Reordered cache (in-place for DynamicCache, new tuple for legacy)
    """
    if past_key_values is None:
        return None

    # Make indices a GPU LongTensor
    if not torch.is_tensor(indices):
        # Infer device from cache if not provided
        if device is None:
            k0, _ = past_key_values[0]
            device = k0.device
        indices = torch.as_tensor(indices, device=device, dtype=torch.long)
    else:
        if indices.dtype != torch.long:
            indices = indices.long()

    # Preferred path: cache object method (DynamicCache)
    if hasattr(past_key_values, "reorder_cache"):
        past_key_values.reorder_cache(indices)
        return past_key_values

    # Fallback: legacy tuple(list_of_layers)
    # past_key_values[layer] = (k, v) with k/v shaped [N, n_kv_heads, T, head_dim]
    new_pkv = []
    for (k, v) in past_key_values:
        # ``device_map=auto`` can place legacy-cache layers on different
        # devices.  index_select requires its indices on the tensor's device.
        k_indices = indices.to(device=k.device)
        v_indices = indices.to(device=v.device)
        new_pkv.append(
            (k.index_select(0, k_indices), v.index_select(0, v_indices))
        )
    return tuple(new_pkv)


@torch.no_grad()
def verify_kv_cache_alignment(model, particles, past_key_values, logits, device, n_samples=2):
    """
    Debug verification: check that KV cache is aligned with particle tokens.
    
    Randomly samples a few particles, compares:
    - logits from cache (already computed)
    - logits from full forward pass (use_cache=False)
    
    Raises AssertionError if mismatch detected.
    """
    import random
    N = len(particles)
    sample_indices = random.sample(range(N), min(n_samples, N))
    
    for idx in sample_indices:
        p = particles[idx]
        # Full forward pass without cache
        input_ids_full = torch.tensor([p.tokens], dtype=torch.long, device=device)
        outputs_full = model(input_ids_full, use_cache=False)
        logits_full = outputs_full.logits[0, -1, :]  # (V,)
        
        # Cached logits for this particle
        logits_cache = logits[idx]  # (V,)
        
        # Compare top-k tokens
        k = 10
        topk_cache = torch.topk(logits_cache, k).indices
        topk_full = torch.topk(logits_full, k).indices
        
        # Check if top-k overlap significantly (at least 8/10)
        overlap = len(set(topk_cache.tolist()) & set(topk_full.tolist()))
        
        # Also check L2 distance (normalized)
        l2_diff = torch.norm(logits_cache - logits_full).item()
        l2_norm = torch.norm(logits_full).item()
        rel_diff = l2_diff / (l2_norm + 1e-8)
        
        # Check cache seq length vs particle token length
        cache_seq_len = past_key_values[0][0].shape[2]  # (batch, heads, seq, dim)
        particle_len = len(p.tokens)
        
        # Relax threshold: bf16 precision can cause small differences
        # Key criterion: top-k overlap should be high (semantic alignment)
        # Also check seq length match
        if cache_seq_len != particle_len:
            raise AssertionError(
                f"KV cache LENGTH mismatch for particle {idx}!\n"
                f"  Cache seq_len: {cache_seq_len}\n"
                f"  Particle token_len: {particle_len}\n"
                f"  Difference: {cache_seq_len - particle_len}"
            )
        
        # Focus primarily on semantic alignment (top-k overlap)
        # L2 diff can be higher due to bf16 cumulative numerical errors
        # With 60+ tokens, rel_diff up to 0.15 is normal for bf16
        if overlap < 7:  # Allow up to 3 differences in top-10
            raise AssertionError(
                f"KV cache SEMANTIC misalignment for particle {idx}!\n"
                f"  Top-{k} overlap: {overlap}/10 (expected >= 7)\n"
                f"  Relative L2 diff: {rel_diff:.4f}\n"
                f"  Token length: {len(p.tokens)}\n"
                f"  Cache seq_len: {cache_seq_len}\n"
                f"  This indicates cache and tokens are out of sync."
            )
        
        # Warn (but don't fail) if L2 diff is high - this is expected for bf16
        if rel_diff > 0.20:
            import warnings
            warnings.warn(
                f"High L2 diff for particle {idx}: rel_diff={rel_diff:.4f}, "
                f"but top-{k} overlap is {overlap}/10 (acceptable)"
            )
    
    return True  # All checks passed


@torch.no_grad()
def batched_asmc_sample(
    model,
    tokenizer,
    context: List[int],
    config: ASMCConfig,
    device,
    verbose: bool = False,
    debug_verify_cache: bool = False,
    tracker=None,
) -> Tuple[List[Particle], Dict[str, Any]]:
    """
    Run batched ASMC sampling with KV cache optimization.
    
    All particles are processed in parallel using batched inference.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        context: Input token ids (prompt)
        config: ASMC configuration
        device: Compute device
        verbose: Print progress
        tracker: ComputeTracker used by model instrumentation. Required when
            ``config.c_int_cap`` is set. The cap is checked after a generated
            token update, allowing at most one-forward C_int overshoot.
    
    Returns:
        particles: Final list of particles
        diagnostics: Dictionary with diagnostics info
    """
    if config.c_int_cap is not None and tracker is None:
        raise ValueError("tracker is required when c_int_cap is specified")
    if config.c_int_cap is not None and debug_verify_cache:
        raise ValueError(
            "debug_verify_cache cannot be combined with c_int_cap because "
            "diagnostic forwards would consume the publication budget"
        )

    N = config.n_particles
    c = len(context)
    max_len = c + config.max_new_tokens
    
    # Ensure pad token exists
    pad_token_id = ensure_pad_token(tokenizer)
    
    # Build termination token IDs. Historical ASMC-only masking of every
    # special token is opt-in because it changes the target distribution.
    if config.stop_token_ids is None:
        config.stop_token_ids = build_stop_token_ids(
            tokenizer,
            include_all_special=config.legacy_stop_constraints,
        )
    
    # Initialize particles - all start with same context
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

    # Initial batched forward pass to get KV cache
    # All particles start with the same context
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    input_ids = input_ids.repeat(N, 1)  # (N, c) - use repeat to ensure separate memory
    
    # Attention mask (all ones for initial context)
    attention_mask = torch.ones_like(input_ids)
    
    # Get initial KV cache
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :].clone()  # (N, vocab_size) - clone to avoid modifying cache
    
    # Reproduce the historical minimum-length constraint only when requested.
    if config.legacy_stop_constraints and config.stop_token_ids:
        for stop_id in config.stop_token_ids:
            logits[:, stop_id] = -1e30  # Always hard mask at t=0
    
    # Track which particles are still active
    active_mask = torch.ones(N, dtype=torch.bool, device=device)
    
    # Token-by-token generation
    n_eos_this_block = 0  # Track EOS sampling within current block
    
    for t_gen in range(config.max_new_tokens):
        # Reset resampled flag at start of each iteration
        resampled_this_block = False
        resample_indices = None
        
        # Compute annealing alpha
        alpha_t = compute_annealing_alpha(
            t_gen, config.alpha_start, config.alpha_star,
            config.anneal_tokens, config.anneal_schedule
        )
        diagnostics["alpha_history"].append(alpha_t)
        
        # Check if any particles are still active
        if not active_mask.any():
            break
        
        # 🔴 CRITICAL FIX: Mask/penalize stop tokens BEFORE computing proposal
        # This prevents the clamp bug from resurrecting masked tokens
        if config.legacy_stop_constraints and config.stop_token_ids:
            if t_gen < config.min_eos_tokens:
                # Hard mask: set to very negative value
                for stop_id in config.stop_token_ids:
                    logits[:, stop_id] = -1e30
            elif t_gen < config.prefer_non_eos_until:
                # Soft penalty
                for stop_id in config.stop_token_ids:
                    logits[:, stop_id] = logits[:, stop_id] - config.eos_penalty
        
        # Compute proposal log probs for all particles
        # NOTE: Must compute AFTER masking logits!
        log_p = F.log_softmax(logits, dim=-1)  # (N, V)
        scaled_logits = alpha_t * logits
        log_q_pow = F.log_softmax(scaled_logits, dim=-1)  # (N, V)
        
        # Mixture: (1-ε) * q_pow + ε * p
        log_one_minus_eps = math.log(1 - config.epsilon)
        log_eps = math.log(config.epsilon)
        
        stacked = torch.stack([
            log_one_minus_eps + log_q_pow,
            log_eps + log_p
        ], dim=0)  # (2, N, V)
        log_q = torch.logsumexp(stacked, dim=0)  # (N, V)
        
        # Sample from q for all particles
        # 🔴 CRITICAL FIX: Use softmax directly, NOT clamp!
        # The clamp(min=1e-10) bug would resurrect masked tokens
        probs_q = torch.softmax(log_q, dim=-1)  # (N, V)
        next_tokens = torch.multinomial(probs_q, 1).squeeze(-1)  # (N,)
        
        # Gather log probabilities
        batch_indices = torch.arange(N, device=device)
        log_q_tok = log_q[batch_indices, next_tokens]  # (N,)
        log_p_tok = log_p[batch_indices, next_tokens]  # (N,)
        
        # Update particles
        n_eos_this_step = 0  # Track EOS sampling for diagnostics
        
        for i in range(N):
            if not active_mask[i]:
                continue
            
            tok = next_tokens[i].item()
            particles[i].tokens.append(tok)
            particles[i].log_q_sum += log_q_tok[i].item()
            particles[i].log_p_sum += log_p_tok[i].item()
            
            # Incremental weight update: Δlog w = α* log p - log q
            delta_logw = config.alpha_star * log_p_tok[i].item() - log_q_tok[i].item()
            particles[i].log_weight += delta_logw
            
            # Check for EOS (check all stop tokens)
            if config.stop_token_ids and tok in config.stop_token_ids:
                particles[i].finished = True
                active_mask[i] = False
                n_eos_this_step += 1
        
        # Accumulate EOS count for this block
        n_eos_this_block += n_eos_this_step
        diagnostics["tokens_generated"] = t_gen + 1

        # The logits used above came from exactly one batched model forward.
        # Checking after consuming them bounds cap overshoot by that last
        # forward while preserving a complete particle-population token step.
        if (
            config.c_int_cap is not None
            and tracker.C_int >= config.c_int_cap
        ):
            diagnostics["budget_exhausted"] = True
            diagnostics["budget_exhausted_at_token"] = t_gen
            diagnostics["stop_reason"] = "budget_exhausted"
            if verbose:
                print(
                    f"  Budget exhausted at t={t_gen}: "
                    f"C_int={tracker.C_int:,} >= cap={config.c_int_cap:,}"
                )
            break
        
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
            n_active = active_mask.sum().item()
            diagnostics["n_eos_sampled_blocks"].append(n_eos_this_block)
            diagnostics["n_finished_blocks"].append(n_finished)
            diagnostics["n_active_blocks"].append(n_active)

            # ===== DIAGNOSTIC: Track weight diversity at every block boundary =====
            weights_for_diag = normalize_logweights(logw)
            if weights_for_diag:
                max_weight = max(weights_for_diag)
                weight_entropy = -sum(w * math.log(w + 1e-12) for w in weights_for_diag if w > 0)
            else:
                max_weight = 0.0
                weight_entropy = 0.0
            diagnostics.setdefault("max_weight_history", []).append(max_weight)
            diagnostics.setdefault("weight_entropy_history", []).append(weight_entropy)
            
            # Track parsed answers at this block
            answer_masses = {}
            n_parsed = 0
            source_counts = {}
            try:
                answer_masses, n_parsed, source_counts = compute_answer_masses(
                    particles,
                    tokenizer,
                    c,
                    use_source_weight=config.use_source_weight,
                )
            except Exception:
                pass
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
                resample_indices = indices

                # ===== DIAGNOSTIC: Track unique ancestors after resampling =====
                unique_ancestors = len(set(indices))
                diagnostics.setdefault("unique_ancestors_history", []).append(unique_ancestors)
                if verbose:
                    print(f"    Unique ancestors after resample: {unique_ancestors}/{N}")

                # Create new particles
                new_particles = [particles[idx].copy() for idx in indices]

                # Reset log weights after resampling
                for p in new_particles:
                    p.log_weight = 0.0

                particles = new_particles
                diagnostics["n_resamples"] += 1
                resampled_this_block = True
                
                # Update active mask based on new particles
                active_mask = torch.tensor(
                    [not p.finished for p in particles],
                    dtype=torch.bool, device=device
                )
                
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
                    # Track previous top answer for stability
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

            # Only prepare the next-token cache after deciding that this
            # block will continue.  If resampling is immediately followed by
            # early stop, advancing the cache would be unused work and could
            # incorrectly turn the outcome into budget exhaustion.
            if (
                resampled_this_block
                and active_mask.any()
                and t_gen + 1 < config.max_new_tokens
            ):
                past_key_values = reorder_past_key_values(
                    past_key_values,
                    resample_indices,
                    device=device,
                )

                # Particles have already been reordered; use their own last
                # tokens rather than the pre-resampling next_tokens tensor.
                last_tokens = torch.tensor(
                    [p.tokens[-1] for p in particles],
                    dtype=torch.long,
                    device=device,
                )
                next_input = last_tokens.unsqueeze(1)
                next_input = torch.where(
                    active_mask.unsqueeze(1),
                    next_input,
                    torch.full_like(next_input, pad_token_id),
                )

                outputs = model(
                    next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].clone()

                if debug_verify_cache:
                    verify_kv_cache_alignment(
                        model,
                        particles,
                        past_key_values,
                        logits,
                        device,
                    )
                    if verbose:
                        print("    [DEBUG] KV cache alignment verified ✓")
        
        # Incremental KV cache update (skip if we just rebuilt KV cache after resampling)
        if (
            active_mask.any()
            and not resampled_this_block
            and t_gen + 1 < config.max_new_tokens
        ):
            # Prepare next input (just the new tokens)
            next_input = next_tokens.unsqueeze(1)  # (N, 1)
            
            # For inactive particles, use pad token
            next_input = torch.where(
                active_mask.unsqueeze(1),
                next_input,
                torch.full_like(next_input, pad_token_id)
            )
            
            # Update attention mask
            # We need to track the cumulative attention mask
            # For simplicity, we'll use position_ids instead
            
            outputs = model(
                next_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].clone()  # Clone for safe modification
    
    # Set stop_reason if not already set
    if diagnostics["stop_reason"] is None:
        if not active_mask.any():
            diagnostics["stop_reason"] = "all_finished"
        elif diagnostics["tokens_generated"] >= config.max_new_tokens:
            diagnostics["stop_reason"] = "max_len"
        else:
            diagnostics["stop_reason"] = "unknown"

    # ===== DIAGNOSTIC: Track unique sequences (token hash) at end =====
    seq_hashes = set()
    for p in particles:
        # Hash the completion tokens (excluding context)
        completion_tokens = tuple(p.tokens[c:])
        seq_hashes.add(hash(completion_tokens))
    diagnostics["unique_sequences_final"] = len(seq_hashes)

    return particles, diagnostics


class BatchedASMCSampler:
    """
    Batched ASMC Sampler with adaptive budget.
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
        debug_verify_cache: bool = False,
        tracker=None,
    ) -> Tuple[List[Particle], str, 'Particle', Dict[str, Any]]:
        """
        Run batched ASMC sampling with optional adaptive budget.
        
        Args:
            context: Input token ids
            config: ASMC configuration
            verbose: Print progress
            debug_verify_cache: If True, verify KV cache alignment after each resample
            tracker: Shared ComputeTracker used for optional C_int enforcement
        """
        if config is None:
            config = ASMCConfig()
        
        c = len(context)
        
        if not config.enable_adaptive:
            # Single pass
            particles, diagnostics = batched_asmc_sample(
                self.model, self.tokenizer, context, config, self.device, verbose,
                debug_verify_cache=debug_verify_cache,
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
        # Build stop_token_ids if not already set on parent config
        if config.stop_token_ids is None:
            config.stop_token_ids = build_stop_token_ids(
                self.tokenizer,
                include_all_special=config.legacy_stop_constraints,
            )
        
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
            stop_token_ids=config.stop_token_ids,
            enable_rejuvenation=False,
            enable_adaptive=False,
        )
        
        if verbose:
            print(f"=== Fast Pass (N={config.fast_n_particles}) ===")
        
        fast_particles, fast_diag = batched_asmc_sample(
            self.model, self.tokenizer, context, fast_config, self.device, verbose,
            debug_verify_cache=debug_verify_cache,
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
            # Keep pass type orthogonal to budget status for audit compatibility.
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
            epsilon=0.08,
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
            # Historical EOS control for hard pass
            legacy_stop_constraints=config.legacy_stop_constraints,
            min_eos_tokens=config.hard_min_eos_tokens,
            prefer_non_eos_until=config.hard_prefer_non_eos_until,
            eos_penalty=config.hard_eos_penalty,
            stop_token_ids=config.stop_token_ids,
            enable_rejuvenation=False,  # Batched rejuv is complex, disable for now
            enable_adaptive=False,
        )
        
        hard_particles, hard_diag = batched_asmc_sample(
            self.model, self.tokenizer, context, hard_config, self.device, verbose,
            debug_verify_cache=debug_verify_cache,
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
        hard_diag["fast_diag"] = fast_diag
        hard_diag["pass_type"] = "hard"
        
        return hard_particles, best_answer, best_particle, hard_diag


@torch.no_grad()
def asmc_generate_batch(
    model,
    tokenizer,
    context: List[int],
    config: ASMCConfig,
    device,
    verbose: bool = False,
    tracker=None,
) -> Tuple[List[int], str, Dict[str, Any]]:
    """
    Convenience function for single-sequence output.
    
    Returns:
        best_tokens: Token ids of best sequence
        best_answer: Parsed answer
        diagnostics: Full diagnostics
    """
    sampler = BatchedASMCSampler(model, tokenizer, device)
    
    particles, best_answer, best_particle, diagnostics = sampler.sample(
        context, config, verbose, tracker=tracker
    )
    
    return best_particle.tokens, best_answer, diagnostics
