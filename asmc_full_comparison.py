#!/usr/bin/env python
"""Run ASMC on the bundled MATH500 benchmark.

ASMC is the only method enabled by default.  Historical comparison baselines
remain available as explicit, unsupported research utilities; they are not
part of the ASMC-only public reproduction contract.  The methods do not all
target the same trajectory distribution.  In
particular, token-wise low-temperature decoding is retained as a diagnostic
baseline and must not be described as sampling the globally power-shaped
trajectory distribution ``p(x) ** alpha``.  ASMC and the sequential MCMC
baseline are the methods intended to approximate that trajectory-level target.
"""

import os
import sys
import json
import hashlib
import importlib.metadata
import math
import random
import argparse
import subprocess
from datetime import datetime, timezone
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import transformers


RNG_PROTOCOL = "sha256-canonical-json-u32-method-isolation-v1"
DEFAULT_MODEL_REVISIONS = {
    "Qwen/Qwen2.5-Math-7B": "8daf1d676c3f24ddec5a99c5cff00a5c0e1c441c",
}
METHOD_PROTOCOLS = {
    "greedy": "deterministic-greedy-decoding-v2",
    "asmc": "cache-coherent-asmc-corrected-v1",
    "naive": "single-temperature-sample-v2",
    "std": "single-temperature-one-sample-v2",
    "mcmc": "completion-only-eos-mcmc-power-sampling-v4",
    "majority": "independent-sampling-unweighted-answer-majority-v2",
    "bestofn": (
        "independent-generation-unconditional-length-normalized-"
        "logprob-argmax-v3"
    ),
}

try:  # Package import
    from .grader_utils.parse_utils import parse_answer_robust
    from .grader_utils.math_grader import grade_answer
    from .constants import PROMPT, COT, BASE
    from .compute_tracker import ComputeTracker
    from .compute_instrumentation import instrument_model
except ImportError:  # Direct script execution from the repository root
    from grader_utils.parse_utils import parse_answer_robust
    from grader_utils.math_grader import grade_answer
    from constants import PROMPT, COT, BASE
    from compute_tracker import ComputeTracker
    from compute_instrumentation import instrument_model


_BASELINE_SUPPORT = None


def _load_baseline_support():
    """Load optional comparison-generation helpers only when requested."""
    global _BASELINE_SUPPORT
    if _BASELINE_SUPPORT is None:
        try:  # Package import
            from . import bestofn as baseline_support
        except ImportError:  # Direct script execution from the repository root
            import bestofn as baseline_support
        _BASELINE_SUPPORT = baseline_support
    return _BASELINE_SUPPORT


def resolved_sampling_generation_kwargs(tokenizer):
    """Resolve optional baseline generation settings lazily."""
    return _load_baseline_support().resolved_sampling_generation_kwargs(
        tokenizer
    )


def resolved_sampling_protocol_metadata(tokenizer):
    """Resolve optional baseline provenance settings lazily."""
    return _load_baseline_support().resolved_sampling_protocol_metadata(
        tokenizer
    )


def resolve_model_revision(model_id, requested_revision):
    """Resolve a reproducible public default without guessing other models."""
    if requested_revision:
        return requested_revision
    return DEFAULT_MODEL_REVISIONS.get(model_id)


def format_prompt(question, model_name, tokenizer, cot=True):
    """Format prompt based on model type."""
    if model_name in ["qwen", "qwen_math"]:
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE
    elif model_name in ["qwen_math_grpo", "phi_grpo", "phi", "tulu"]:
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )
    else:
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE
    return format_str


class AutoregressiveSampler:
    """Wrapper for autoregressive sampling with log probability access."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    @torch.no_grad()
    def next_token(self, prefix):
        """Return log probs for next token."""
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=self.device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits[0, -1, :]
        return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def naive_temp_sample(sampler, context, temp, max_new_tokens):
    """Draw one token-wise low-temperature autoregressive sample.

    This changes each normalized next-token conditional separately; it does
    not sample from the globally power-shaped trajectory distribution.
    
    Returns: (tokens, completion_text)
    """
    device = sampler.device
    tokenizer = sampler.tokenizer
    model = sampler.model
    
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        return_dict_in_generate=True,
        temperature=temp,
        **resolved_sampling_generation_kwargs(tokenizer),
    )
    
    generated_ids = output.sequences[0][len(context):].tolist()
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_ids, completion


@torch.no_grad()
def greedy_sample(sampler, context, max_new_tokens):
    """Deterministic greedy baseline used in the paper tables."""
    input_ids = torch.tensor(
        [context], dtype=torch.long, device=sampler.device
    )
    output = sampler.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        **resolved_sampling_generation_kwargs(sampler.tokenizer),
    )
    generated_ids = output.sequences[0][len(context):].tolist()
    completion = sampler.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    return generated_ids, completion


@torch.no_grad()
def std_sample(sampler, context, max_new_tokens):
    """
    Standard sampling: sample from p (temperature=1.0, default)
    This is the baseline without any temperature scaling.
    
    Returns: (tokens, completion_text)
    """
    device = sampler.device
    tokenizer = sampler.tokenizer
    model = sampler.model
    
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    
    # Temperature 1 leaves the logits unchanged.  Termination, padding, cache
    # behavior, and neutral warpers are all pinned explicitly below.
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        return_dict_in_generate=True,
        **resolved_sampling_generation_kwargs(tokenizer),
    )
    
    generated_ids = output.sequences[0][len(context):].tolist()
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_ids, completion


@torch.no_grad()
def naive_majority_vote(sampler, context, temp, max_new_tokens, n_samples=64):
    """
    Naive temperature sampling plus majority voting.
    
    Generate N independent samples with temperature sampling, parse answers,
    and return the majority vote answer.
    
    Matching the number of samples to the particle count does not, by itself,
    make this baseline compute-matched.  Use the audited realized-compute
    selector for paper comparisons.
    
    Args:
        sampler: AutoregressiveSampler
        context: Input token ids
        temp: Temperature (alpha = 1/temp)
        max_new_tokens: Max tokens to generate
        n_samples: Number of samples (should match ASMC n_particles)
    
    Returns:
        best_completion: Completion from the sample with majority answer
        best_answer: The majority vote answer
        vote_info: Dict with voting statistics
    """
    from collections import Counter
    
    device = sampler.device
    tokenizer = sampler.tokenizer
    model = sampler.model
    
    completions = []
    completion_token_ids = []
    answers = []
    
    # Generate N samples
    for _ in range(n_samples):
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            return_dict_in_generate=True,  # Fix: need this to access output.sequences
            **resolved_sampling_generation_kwargs(tokenizer),
        )
        
        generated_ids = output.sequences[0][len(context):].tolist()
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)
        completion_token_ids.append(generated_ids)
        
        # Parse answer
        answer = parse_answer_robust(completion)
        answers.append(answer)
    
    # Count valid answers (non-None)
    valid_answers = [(i, a) for i, a in enumerate(answers) if a is not None]
    n_valid = len(valid_answers)
    
    if n_valid == 0:
        # No valid answers, return first completion
        return completions[0], None, {
            "n_samples": n_samples,
            "n_valid": 0,
            "n_unique": 0,
            "best_count": 0,
            "best_mass": 0.0,
            "selected_token_ids": completion_token_ids[0],
        }
    
    # Majority voting
    answer_counts = Counter(str(a) for _, a in valid_answers)
    best_answer_str, best_count = answer_counts.most_common(1)[0]
    
    # Find the first sample with this answer
    best_idx = None
    best_answer = None
    for idx, ans in valid_answers:
        if str(ans) == best_answer_str:
            best_idx = idx
            best_answer = ans
            break
    
    vote_info = {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "n_unique": len(answer_counts),
        "best_count": best_count,
        "best_mass": best_count / n_samples,
        "selected_token_ids": completion_token_ids[best_idx],
    }
    
    return completions[best_idx], best_answer, vote_info


def _mcmc_log_ratio_terms(
    current_q_log_probs,
    current_target_log_probs,
    proposed_q_log_probs,
    proposed_target_log_probs,
    truncation_offset,
    current_n_tokens,
    proposed_n_tokens,
):
    """Return the three MH log-ratio terms for a suffix proposal.

    For generated sequence ``x`` and proposal ``y``, the target is
    ``pi(x) proportional to p(x) ** alpha``.  A proposal first chooses one of
    the generated-token positions uniformly, then samples the replacement
    suffix from ``q``.  After the common prefix cancels, the MH log ratio is

    ``target_delta + suffix_proposal_delta + truncation_choice_delta``.

    The final term is ``log(n_x / n_y)`` because the forward and reverse
    truncation positions have probabilities ``1 / n_x`` and ``1 / n_y``.
    """
    if current_n_tokens <= 0 or proposed_n_tokens <= 0:
        raise ValueError("MCMC states must contain at least one generated token")
    if not 0 <= truncation_offset < current_n_tokens:
        raise ValueError("MCMC truncation offset is outside the current state")
    if truncation_offset >= proposed_n_tokens:
        raise ValueError("MCMC proposal does not extend beyond its prefix")
    if (
        len(current_q_log_probs) != current_n_tokens
        or len(current_target_log_probs) != current_n_tokens
    ):
        raise ValueError("MCMC current-state log probabilities are misaligned")
    expected_proposed_suffix = proposed_n_tokens - truncation_offset
    if (
        len(proposed_q_log_probs) != expected_proposed_suffix
        or len(proposed_target_log_probs) != expected_proposed_suffix
    ):
        raise ValueError("MCMC proposal log probabilities are misaligned")

    # The complete current suffix is required here.  Clipping it to the
    # proposal length would drop the probability of a tail removed by EOS.
    current_q_suffix = current_q_log_probs[truncation_offset:]
    current_target_suffix = current_target_log_probs[truncation_offset:]
    return {
        "target": (
            sum(proposed_target_log_probs) - sum(current_target_suffix)
        ),
        "proposal": sum(current_q_suffix) - sum(proposed_q_log_probs),
        "truncation_choice": (
            math.log(current_n_tokens) - math.log(proposed_n_tokens)
        ),
    }


def _mcmc_log_acceptance_ratio(
    current_q_log_probs,
    current_target_log_probs,
    proposed_q_log_probs,
    proposed_target_log_probs,
    truncation_offset,
    current_n_tokens,
    proposed_n_tokens,
    horizon_n_tokens,
):
    """Return the MH log ratio under a fixed block-level length horizon.

    Both directions regenerate from the selected prefix to the same fixed
    horizon.  They may stop earlier on EOS, so shorter and longer states have
    bidirectional support as long as both lie within that horizon.  The
    state-dependent number of truncation choices is handled by the
    ``log(n_x / n_y)`` term from :func:`_mcmc_log_ratio_terms`.
    """
    if horizon_n_tokens <= 0:
        raise ValueError("MCMC proposal horizon must be positive")
    if max(current_n_tokens, proposed_n_tokens) > horizon_n_tokens:
        raise ValueError("MCMC state exceeds the fixed proposal horizon")
    terms = _mcmc_log_ratio_terms(
        current_q_log_probs,
        current_target_log_probs,
        proposed_q_log_probs,
        proposed_target_log_probs,
        truncation_offset,
        current_n_tokens,
        proposed_n_tokens,
    )
    return sum(terms.values())


def _mcmc_generated_token_log_probs(unscaled_logits, tokens, temp):
    """Score generated tokens under proposal q and unnormalised p**alpha."""
    if temp <= 0:
        raise ValueError("MCMC temperature must be positive")
    idx = tokens.view(-1, 1, 1)
    target_log_probs = (
        (1.0 / temp)
        * torch.gather(
            F.log_softmax(unscaled_logits, dim=-1), -1, idx
        )
    ).view(-1).tolist()
    proposal_log_probs = torch.gather(
        F.log_softmax(unscaled_logits / temp, dim=-1), -1, idx
    ).view(-1).tolist()
    return proposal_log_probs, target_log_probs


@torch.no_grad()
def mcmc_power_sample(sampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    """
    MCMC Power Sampling: targets p^alpha using autoregressive MCMC.
    
    Algorithm:
    1. Generate initial sequence with temperature sampling
    2. For each MCMC step:
       - Pick random position
       - Propose new continuation from that position
       - Accept/reject based on MH ratio
    
    Returns: (tokens, completion_text, acceptance_ratio)
    """
    device = sampler.device
    tokenizer = sampler.tokenizer
    model = sampler.model
    c = len(context)
    
    gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []
    
    if block_num < 1:
        raise ValueError("block_num must be positive")
    # The requested block count is part of the method identity.  Silently
    # falling back to one block would make result metadata false.
    if max_new_tokens % block_num != 0:
        raise ValueError("max_new_tokens must be divisible by block_num")
    jump_size = max_new_tokens // block_num
    
    attempts = 0
    acceptances = 0
    
    def generate_with_temp(prefix, seq_len, temp):
        """Generate sequence with temperature and return log probs."""
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        n_new = seq_len - len(prefix)
        if n_new <= 0:
            return prefix, [], []
            
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=n_new,
            do_sample=True,
            temperature=temp,
            return_dict_in_generate=True,
            output_logits=True,
            **resolved_sampling_generation_kwargs(tokenizer),
        )
        
        prop = output.sequences[0].tolist()
        
        if len(output.logits) == 0:
            return prop, [], []
            
        unscaled_logits = torch.stack(output.logits, dim=0)
        tokens = output.sequences[0][len(prefix):]
        
        if len(tokens) != unscaled_logits.shape[0]:
            # Handle mismatch
            return prop, [], []
        
        log_probs_norm, log_probs_unnorm = _mcmc_generated_token_log_probs(
            unscaled_logits,
            tokens,
            temp,
        )
        
        return prop, log_probs_norm, log_probs_unnorm
    
    # Generate blocks with MCMC refinement
    for block_idx in range(block_num):
        # Stage horizons advance absolutely to the requested final T even if a
        # prior state terminated early.  A state that already contains a
        # generated EOS is a valid short trajectory at the new horizon: do not
        # append after EOS.  It can only be revised or lengthened by a valid MH
        # truncation proposal below.
        target_len = c + (block_idx + 1) * jump_size
        generated = gen[c:]
        has_generated_eos = tokenizer.eos_token_id in generated
        if not has_generated_eos:
            gen, lp_norm, lp_unnorm = generate_with_temp(
                gen, target_len, temp
            )
            log_probs_norm.extend(lp_norm)
            log_probs_unnorm.extend(lp_unnorm)
        
        # MCMC steps within block
        for _ in range(mcmc_steps):
            if len(gen) <= c:
                break
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)
            
            # Propose to the block's fixed horizon, not the current state's
            # length.  This makes an EOS-shortened state reversible: a later
            # proposal can grow back toward the same horizon.
            prop, log_prob_prop, target_log_prob_prop = generate_with_temp(
                gen[:idx], target_len, temp
            )
            s = len(prop)
            
            if len(log_prob_prop) == 0 or s <= idx:
                continue
                
            start_idx = idx - c

            if start_idx < 0:
                continue

            log_r = _mcmc_log_acceptance_ratio(
                current_q_log_probs=log_probs_norm,
                current_target_log_probs=log_probs_unnorm,
                proposed_q_log_probs=log_prob_prop,
                proposed_target_log_probs=target_log_prob_prop,
                truncation_offset=start_idx,
                current_n_tokens=t - c,
                proposed_n_tokens=s - c,
                horizon_n_tokens=target_len - c,
            )

            # Avoid overflow for positive log ratios.
            if math.isnan(log_r) or log_r == -math.inf:
                continue
            if log_r >= 0.0 or np.random.rand() < math.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                # Update log probs
                if idx - c < len(log_probs_norm):
                    log_probs_norm[idx-c:] = log_prob_prop
                    log_probs_unnorm[idx-c:] = target_log_prob_prop
        
        # Canonicalize at the first generated EOS, but continue to every later
        # absolute stage horizon.  Prompt tokens must never truncate the
        # completion or its generation-relative log-probability arrays.
        generated = gen[c:]
        if tokenizer.eos_token_id in generated:
            eos_rel_idx = generated.index(tokenizer.eos_token_id)
            n_generated_kept = eos_rel_idx + 1
            gen = gen[:c + n_generated_kept]
            log_probs_norm = log_probs_norm[:n_generated_kept]
            log_probs_unnorm = log_probs_unnorm[:n_generated_kept]
    
    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    full_sequence = gen  # includes context for compatibility with callers
    completion = tokenizer.decode(gen[c:], skip_special_tokens=True)
    
    return full_sequence, completion, acceptance_ratio


def _record_completion_evidence(
    result, method, token_ids, eos_token_id
):
    """Persist lossless generated-token evidence for one selected completion."""
    if token_ids is None:
        result[f"{method}_completion_token_ids"] = None
        result[f"{method}_completion_has_eos"] = None
        return

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().view(-1).tolist()
    canonical_ids = [int(token_id) for token_id in token_ids]
    if isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(token_id) for token_id in eos_token_id}
    elif eos_token_id is None:
        eos_ids = set()
    else:
        eos_ids = {int(eos_token_id)}
    result[f"{method}_completion_token_ids"] = json.dumps(
        canonical_ids,
        separators=(",", ":"),
    )
    result[f"{method}_completion_has_eos"] = any(
        token_id in eos_ids for token_id in canonical_ids
    )


def _add_paper_compute_fields(result, method, compute=None, time_s=None):
    """Add unambiguous paper-metric columns while retaining legacy columns."""
    if compute is None:
        result[f"{method}_c_int"] = None
        result[f"{method}_c_tok"] = None
        result[f"{method}_c_step"] = None
        result[f"{method}_n_forward"] = None
    else:
        result[f"{method}_c_int"] = compute["C_int"]
        result[f"{method}_c_tok"] = compute["C_tok"]
        result[f"{method}_c_step"] = compute["C_step"]
        result[f"{method}_n_forward"] = compute["forward_calls"]
        if time_s is None:
            time_s = compute["total_time"]
    result[f"{method}_time_s"] = time_s


def _derive_method_rng_metadata(
    base_seed, problem_idx, method, method_identity
):
    """Derive a stable, auditable RNG seed for one problem/method pair.

    The seed key deliberately excludes the enabled-method set and every other
    method's configuration.  Consequently, consuming randomness in one method
    cannot perturb a later method, while changing a method's own effective
    protocol gives it a distinct stream.
    """
    key = {
        "base_seed": int(base_seed),
        "method": str(method),
        "method_identity": method_identity,
        "problem_idx": int(problem_idx),
        "rng_protocol": RNG_PROTOCOL,
    }
    key_payload = json.dumps(
        key,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    key_sha256 = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
    # A common uint32 seed is accepted without remapping by Python, NumPy, and
    # Torch.  The complete digest remains in the row for collision auditing.
    rng_seed = int(key_sha256[:8], 16)
    return {
        "protocol": RNG_PROTOCOL,
        "seed": rng_seed,
        "key_sha256": key_sha256,
        "key_payload": key_payload,
    }


def _seed_all_rngs(seed):
    """Reset every global RNG used by the comparison implementations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _activate_method_rng(
    result, base_seed, problem_idx, method, method_identity
):
    """Record and activate a method-local random stream for a result row."""
    metadata = _derive_method_rng_metadata(
        base_seed, problem_idx, method, method_identity
    )
    result[f"{method}_rng_protocol"] = metadata["protocol"]
    result[f"{method}_rng_seed"] = metadata["seed"]
    result[f"{method}_rng_key_sha256"] = metadata["key_sha256"]
    result[f"{method}_rng_key_payload"] = metadata["key_payload"]
    _seed_all_rngs(metadata["seed"])
    return metadata


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository):
    """Return explicit Git provenance without inventing a clean code SHA."""
    try:
        commit = subprocess.run(
            ["git", "-C", repository, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", repository, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"git_commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "dirty": None}


def _package_version_or_status(distribution):
    """Return an installed distribution version or an explicit status."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


def _nvidia_driver_version():
    """Return the NVIDIA driver version without inventing unavailable data."""
    if not torch.cuda.is_available():
        return "not-applicable"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        versions = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
        return ",".join(versions) if versions else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _completion_status(method_error_rows):
    """Return manifest status and process exit code for method errors."""
    if any(method_error_rows.values()):
        return "completed_with_errors", 1
    return "complete", 0


def _write_manifest(path, manifest):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _write_results_csv(path, rows):
    """Atomically checkpoint complete result rows."""
    temporary_path = f"{path}.tmp"
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed or adaptive ASMC on MATH500; optional comparison "
            "baselines are opt-in research utilities"
        )
    )
    
    # Basic settings
    parser.add_argument("--save_str", type=str, default="results/", help="Save directory")
    parser.add_argument("--model", type=str, default="qwen_math",
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help=(
            "Hugging Face model revision/commit (Qwen2.5-Math-7B uses the "
            "documented immutable release default when omitted)"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow model repositories to execute custom Python code",
    )
    parser.add_argument("--dataset", type=str, default="MATH", help="Dataset name")
    cot_group = parser.add_mutually_exclusive_group()
    cot_group.add_argument(
        "--cot", dest="cot", action="store_true", help="Use chain-of-thought prompting (default)"
    )
    cot_group.add_argument(
        "--no_cot", dest="cot", action="store_false", help="Disable chain-of-thought prompting"
    )
    parser.set_defaults(cot=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device for model placement (default: Transformers device_map=auto)",
    )
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index (0-4)")
    parser.add_argument("--n_problems", type=int, default=None, help="Limit number of problems (for sanity check)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_tokens", type=int, default=3072, help="Max generation tokens")
    
    # Temperature/alpha setting (shared across methods)
    # Use --temp to match gold standard (power_samp_math_csd3.sh)
    parser.add_argument("--temp", type=float, default=0.25, dest="temperature", help="Temperature (alpha = 1/temp)")
    
    # MCMC settings
    parser.add_argument("--mcmc_steps", type=int, default=10, help="MCMC steps per block")
    parser.add_argument("--mcmc_blocks", type=int, default=16, help="Number of MCMC blocks")
    
    # ASMC settings
    parser.add_argument("--alpha_star", type=float, default=None, help="Target alpha (default: 1/temperature)")
    parser.add_argument(
        "--c_int_cap", "--cint_cap",
        dest="c_int_cap",
        type=float,
        default=None,
        help=(
            "Per-problem ASMC C_int cap. It is checked after each "
            "forward-backed generation update, so usage may overshoot by at "
            "most the final model forward"
        ),
    )
    parser.add_argument("--n_particles", type=int, default=64, help="Number of particles")
    parser.add_argument("--block_size", type=int, default=32, help="Block size for ESS check")
    parser.add_argument("--ess_threshold", type=float, default=0.5, help="ESS threshold for resampling")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Defensive mixture epsilon")
    parser.add_argument("--anneal_tokens", type=int, default=512, help="Annealing duration (tokens)")
    parser.add_argument("--alpha_start", type=float, default=1.5, help="Starting alpha")
    parser.add_argument(
        "--hard_anneal_tokens",
        type=int,
        default=768,
        help="Adaptive hard-pass annealing duration (tokens)",
    )
    parser.add_argument(
        "--hard_alpha_start",
        type=float,
        default=1.3,
        help="Adaptive hard-pass starting alpha",
    )
    parser.add_argument(
        "--hard_ess_threshold",
        type=float,
        default=0.6,
        help="Adaptive hard-pass ESS resampling threshold",
    )
    parser.add_argument("--anneal_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--early_stop_mass", type=float, default=0.80, help="Early stop mass threshold")
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument(
        "--adaptive", "--enable_adaptive",
        dest="enable_adaptive",
        action="store_true",
        help="Use the adaptive fast/hard ASMC policy",
    )
    adaptive_group.add_argument(
        "--fixed", "--no_adaptive",
        dest="enable_adaptive",
        action="store_false",
        help="Use a single fixed-N ASMC pass (default)",
    )
    parser.set_defaults(enable_adaptive=False)
    parser.add_argument("--fast_mass_threshold", type=float, default=0.65, help="Fast pass mass threshold")
    parser.add_argument(
        "--hard_n_particles",
        type=int,
        default=None,
        help="Hard-pass particles (default: --n_particles)",
    )
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--batched", "--use_batched",
        dest="use_batched",
        action="store_true",
        help="Use cache-coherent batched inference (default)",
    )
    backend_group.add_argument(
        "--sequential", "--no_batched",
        dest="use_batched",
        action="store_false",
        help="Use the reference sequential ASMC implementation",
    )
    parser.set_defaults(use_batched=True)
    parser.add_argument(
        "--legacy_stop_constraints",
        action="store_true",
        default=False,
        help=(
            "Re-enable the historical ASMC-only stop-token masking and EOS "
            "penalty; disabled by default for method-comparable decoding"
        ),
    )
    
    # Method selection
    run_asmc_group = parser.add_mutually_exclusive_group()
    run_asmc_group.add_argument(
        "--run_asmc", dest="run_asmc", action="store_true", help="Run ASMC (default)"
    )
    run_asmc_group.add_argument(
        "--no_asmc", dest="run_asmc", action="store_false", help="Do not run ASMC"
    )
    parser.set_defaults(run_asmc=True)
    parser.add_argument("--run_greedy", action="store_true", help="Run deterministic greedy decoding")
    parser.add_argument("--run_naive", action="store_true", help="Run naive temp sampling (single sample)")
    parser.add_argument("--run_std", action="store_true", help="Run standard sampling (single sample)")
    parser.add_argument("--run_mcmc", action="store_true", help="Run MCMC power sampling")
    parser.add_argument("--run_majority", action="store_true", help="Run naive temp + majority voting (N samples)")
    parser.add_argument(
        "--run_bestofn",
        action="store_true",
        help=(
            "Run the paper Best-of-N baseline (batched generation plus "
            "length-normalized teacher-forcing reranking)"
        ),
    )
    parser.add_argument(
        "--bestofn_n",
        type=int,
        default=4,
        help="Number of independently generated Best-of-N candidates",
    )
    parser.add_argument(
        "--bestofn_temp",
        type=float,
        default=None,
        help="Best-of-N generation temperature (default: --temp)",
    )
    parser.add_argument(
        "--bestofn_chunk_size",
        type=int,
        default=8,
        help="Initial generation/scoring micro-batch size; halves after CUDA OOM",
    )

    # ASMC voting mode for ablation study
    parser.add_argument("--asmc_vote_mode", type=str, default="weighted_no_source",
        choices=["weighted", "weighted_no_source", "majority", "majority_no_source"],
        help=(
            "ASMC voting strategy: weighted_no_source matches the paper "
            "(default); weighted retains legacy parser-source multipliers"
        ))
    
    # Verbose
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()

    finite_arguments = (
        ("--temp", args.temperature),
        ("--ess_threshold", args.ess_threshold),
        ("--epsilon", args.epsilon),
        ("--alpha_start", args.alpha_start),
        ("--hard_alpha_start", args.hard_alpha_start),
        ("--hard_ess_threshold", args.hard_ess_threshold),
        ("--early_stop_mass", args.early_stop_mass),
        ("--fast_mass_threshold", args.fast_mass_threshold),
    )
    for flag, value in finite_arguments:
        if not math.isfinite(value):
            parser.error(f"{flag} must be finite")

    if not 0 <= args.batch_idx <= 4:
        parser.error("--batch_idx must be between 0 and 4 for MATH500")
    if args.n_problems is not None and args.n_problems <= 0:
        parser.error("--n_problems must be positive")
    if args.max_tokens <= 0:
        parser.error("--max_tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temp must be positive")
    if args.c_int_cap is not None:
        if not math.isfinite(args.c_int_cap):
            parser.error("--c_int_cap must be finite")
        if args.c_int_cap <= 0:
            parser.error("--c_int_cap must be positive")
        if not args.use_batched:
            parser.error("--c_int_cap requires the cache-coherent --batched backend")
    if args.bestofn_n <= 0:
        parser.error("--bestofn_n must be positive")
    if args.bestofn_chunk_size <= 0:
        parser.error("--bestofn_chunk_size must be positive")
    if args.bestofn_temp is None:
        args.bestofn_temp = args.temperature
    if not math.isfinite(args.bestofn_temp):
        parser.error("--bestofn_temp must be finite")
    if args.bestofn_temp <= 0:
        parser.error("--bestofn_temp must be positive")
    if args.n_particles <= 0:
        parser.error("--n_particles must be positive")
    if args.hard_n_particles is not None and args.hard_n_particles <= 0:
        parser.error("--hard_n_particles must be positive")
    if (
        args.hard_n_particles is not None
        and args.hard_n_particles < max(1, args.n_particles // 2)
    ):
        parser.error(
            "--hard_n_particles must be at least the resolved fast-pass "
            "population (N/2)"
        )
    if args.block_size <= 0:
        parser.error("--block_size must be positive")
    if not 0.0 <= args.ess_threshold <= 1.0:
        parser.error("--ess_threshold must be between 0 and 1")
    if not 0.0 < args.epsilon < 1.0:
        parser.error("--epsilon must be strictly between 0 and 1")
    if args.anneal_tokens < 0:
        parser.error("--anneal_tokens must be non-negative")
    if args.alpha_start <= 0:
        parser.error("--alpha_start must be positive")
    if args.hard_anneal_tokens < 0:
        parser.error("--hard_anneal_tokens must be non-negative")
    if args.hard_alpha_start <= 0:
        parser.error("--hard_alpha_start must be positive")
    if not 0.0 <= args.hard_ess_threshold <= 1.0:
        parser.error("--hard_ess_threshold must be between 0 and 1")
    if not 0.0 <= args.early_stop_mass <= 1.0:
        parser.error("--early_stop_mass must be between 0 and 1")
    if not 0.0 <= args.fast_mass_threshold <= 1.0:
        parser.error("--fast_mass_threshold must be between 0 and 1")
    if args.mcmc_steps < 0:
        parser.error("--mcmc_steps must be non-negative")
    if args.mcmc_blocks <= 0:
        parser.error("--mcmc_blocks must be positive")
    if args.run_mcmc and args.max_tokens % args.mcmc_blocks != 0:
        parser.error("--max_tokens must be divisible by --mcmc_blocks")
    
    # Set alpha_star from temperature if not specified
    if args.alpha_star is None:
        args.alpha_star = 1.0 / args.temperature
    if not math.isfinite(args.alpha_star):
        parser.error("--alpha_star must be finite")
    if args.alpha_star <= 0:
        parser.error("--alpha_star must be positive")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    model_name = args.model
    
    # Create save directory
    save_dir = os.path.join(args.save_str, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Model path mapping
    model_map = {
        "qwen": "Qwen/Qwen2.5-7B",
        "qwen_math": "Qwen/Qwen2.5-Math-7B",
        "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "phi_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-phi3.5_mini_inst-lr5e-7-kl0.00-step150",
        "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    }
    model_str = model_map.get(model_name, model_name)
    args.model_revision = resolve_model_revision(
        model_str, args.model_revision
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "MATH":
        json_file = os.path.join(
            os.path.dirname(__file__), 'data', 'MATH500.json'
        )
        dataset_name = "MATH500"
        with open(json_file, "r", encoding="utf-8") as handle:
            dataset = json.load(handle)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"Dataset loaded: {len(dataset)} problems")
    if 100 * args.batch_idx >= len(dataset):
        parser.error(
            f"--batch_idx {args.batch_idx} selects no rows from a "
            f"{len(dataset)}-problem dataset"
        )
    
    # Load model
    print(f"Loading model: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    comparison_enabled = any(
        (
            args.run_greedy,
            args.run_naive,
            args.run_std,
            args.run_mcmc,
            args.run_majority,
            args.run_bestofn,
        )
    )
    sampling_protocol_metadata = (
        resolved_sampling_protocol_metadata(tokenizer)
        if comparison_enabled
        else {}
    )
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    device_map = "auto" if args.device == "auto" else {"": args.device}
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        revision=args.model_revision,
        torch_dtype=dtype_map[args.dtype],
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    hf_model.eval()
    print("Model loaded successfully")
    model_revision = getattr(hf_model.config, "_commit_hash", None)
    if model_revision is None:
        model_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if model_revision is None:
        model_revision = args.model_revision
    
    # Create autoregressive sampler for baselines
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, hf_model.device)
    
    # Import ASMC sampler
    asmc_sampler = None
    asmc_config = None
    asmc_use_source_weight = args.asmc_vote_mode in {"weighted", "majority"}
    if args.run_asmc:
        try:
            from .asmc_sampler import (
                ASMCConfig,
                ASMCSampler,
                build_stop_token_ids,
                weighted_voting_output,
                unweighted_majority_voting,
            )
            from .asmc_batched import BatchedASMCSampler
        except ImportError:
            from asmc_sampler import (
                ASMCConfig,
                ASMCSampler,
                build_stop_token_ids,
                weighted_voting_output,
                unweighted_majority_voting,
            )
            from asmc_batched import BatchedASMCSampler
        
        asmc_config = ASMCConfig(
            alpha_star=args.alpha_star,
            c_int_cap=args.c_int_cap,
            n_particles=args.n_particles,
            block_size=args.block_size,
            max_new_tokens=args.max_tokens,
            ess_threshold=args.ess_threshold,
            epsilon=args.epsilon,
            anneal_tokens=args.anneal_tokens,
            alpha_start=args.alpha_start,
            anneal_schedule=args.anneal_schedule,
            use_source_weight=asmc_use_source_weight,
            early_stop_mass_threshold=args.early_stop_mass,
            early_stop_min_tokens=64,
            enable_rejuvenation=False,
            enable_adaptive=args.enable_adaptive,
            fast_mass_threshold=args.fast_mass_threshold,
            hard_n_particles=args.hard_n_particles,
            hard_anneal_tokens=args.hard_anneal_tokens,
            hard_alpha_start=args.hard_alpha_start,
            hard_ess_threshold=args.hard_ess_threshold,
            legacy_stop_constraints=args.legacy_stop_constraints,
        )
        # Resolve tokenizer-derived termination IDs before content-addressing
        # the protocol.  Samplers then consume this immutable resolved value
        # instead of mutating a payload after its hash has been recorded.
        asmc_config.stop_token_ids = build_stop_token_ids(
            tokenizer,
            include_all_special=args.legacy_stop_constraints,
        )
        
        if args.use_batched:
            asmc_sampler = BatchedASMCSampler(hf_model, tokenizer, hf_model.device)
        else:
            asmc_sampler = ASMCSampler(hf_model, tokenizer, hf_model.device)
    
    # Results list
    results = []
    
    # Batch range
    start = 100 * args.batch_idx
    end = min(100 * (args.batch_idx + 1), len(dataset))
    # Apply n_problems limit if specified (for sanity check)
    if args.n_problems is not None:
        end = min(start + args.n_problems, end)
    
    # Output file
    # Microseconds prevent concurrent configurations from colliding when they
    # share a result directory, batch, seed, and temperature.
    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    csv_path = os.path.join(
        save_dir,
        f"full_comparison_temp{args.temperature}_batch{args.batch_idx}_seed{args.seed}_{timestamp}.csv"
    )
    manifest_path = os.path.splitext(csv_path)[0] + ".manifest.json"
    run_mode = "adaptive" if args.enable_adaptive else "fixed"
    asmc_protocol_for_id = {
        "backend": "batched" if args.use_batched else "sequential",
        "cot": args.cot,
        "vote_mode": args.asmc_vote_mode,
        "config": (
            json.loads(json.dumps(vars(asmc_config), sort_keys=True))
            if asmc_config is not None else None
        ),
    }
    asmc_protocol_payload = json.dumps(
        asmc_protocol_for_id,
        sort_keys=True,
        separators=(",", ":"),
    )
    asmc_protocol_sha256 = hashlib.sha256(
        asmc_protocol_payload.encode("utf-8")
    ).hexdigest()
    asmc_config_id = (
        f"asmc-{run_mode}-n{args.n_particles}-{args.asmc_vote_mode}-"
        f"{asmc_protocol_sha256[:16]}"
    )
    bestofn_config_id = (
        f"n{args.bestofn_n}_temp{args.bestofn_temp:g}_"
        f"chunk{args.bestofn_chunk_size}_lengthnorm"
    )
    repository = os.path.dirname(os.path.abspath(__file__))
    code_state = _git_state(repository)
    dataset_sha256 = _sha256_file(json_file)
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    nvidia_driver_version = _nvidia_driver_version()
    flash_attn_version = _package_version_or_status("flash-attn")
    rng_common_identity = {
        "attn_implementation": args.attn_implementation,
        "cot": args.cot,
        "dataset_name": dataset_name,
        "dataset_sha256": dataset_sha256,
        "dtype": args.dtype,
        "max_new_tokens": args.max_tokens,
        "model_id": model_str,
        "model_revision": model_revision,
    }
    method_rng_identities = {
        "greedy": {
            "common": rng_common_identity,
            "config": {
                "do_sample": False,
                **sampling_protocol_metadata,
            },
            "protocol": METHOD_PROTOCOLS["greedy"],
        },
        "asmc": {
            "common": rng_common_identity,
            "config_id": asmc_config_id,
            "protocol": METHOD_PROTOCOLS["asmc"],
            "protocol_payload": asmc_protocol_for_id,
        },
        "naive": {
            "common": rng_common_identity,
            "config": {
                "temperature": args.temperature,
                **sampling_protocol_metadata,
            },
            "protocol": METHOD_PROTOCOLS["naive"],
        },
        "std": {
            "common": rng_common_identity,
            "config": {
                "temperature": 1.0,
                **sampling_protocol_metadata,
            },
            "protocol": METHOD_PROTOCOLS["std"],
        },
        "mcmc": {
            "common": rng_common_identity,
            "config": {
                "blocks": args.mcmc_blocks,
                "steps_per_block": args.mcmc_steps,
                "temperature": args.temperature,
                **sampling_protocol_metadata,
            },
            "protocol": METHOD_PROTOCOLS["mcmc"],
        },
        "majority": {
            "common": rng_common_identity,
            "config": {
                "n_samples": args.n_particles,
                "temperature": args.temperature,
                **sampling_protocol_metadata,
            },
            "protocol": METHOD_PROTOCOLS["majority"],
        },
        "bestofn": {
            "common": rng_common_identity,
            "config": {
                "generation_batch_size": args.bestofn_chunk_size,
                "length_normalize": True,
                "n": args.bestofn_n,
                "scoring_batch_size": args.bestofn_chunk_size,
                "temperature": args.bestofn_temp,
                **sampling_protocol_metadata,
            },
            "config_id": bestofn_config_id,
            "protocol": METHOD_PROTOCOLS["bestofn"],
        },
    }
    rng_method_enabled = {
        "greedy": args.run_greedy,
        "asmc": args.run_asmc,
        "naive": args.run_naive,
        "std": args.run_std,
        "mcmc": args.run_mcmc,
        "majority": args.run_majority,
        "bestofn": args.run_bestofn,
    }
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": timestamp,
        "started_at_utc": started_at.isoformat(),
        "code": code_state,
        "model": {
            "id": model_str,
            "revision": model_revision,
            "dtype": args.dtype,
        },
        "data": {
            "name": dataset_name,
            "path": os.path.relpath(json_file, repository),
            "sha256": dataset_sha256,
            "rows": len(dataset),
        },
        "protocol": {
            **vars(args),
            **sampling_protocol_metadata,
            "mode": run_mode,
            "resolved_fast_n_particles": (
                asmc_config.fast_n_particles if asmc_config is not None else None
            ),
            "resolved_hard_n_particles": (
                asmc_config.hard_n_particles if asmc_config is not None else None
            ),
            "resolved_asmc_config": (
                asmc_protocol_for_id["config"]
                if asmc_config is not None else None
            ),
            "asmc_config_id": asmc_config_id if asmc_config is not None else None,
            "asmc_protocol_sha256": (
                asmc_protocol_sha256 if asmc_config is not None else None
            ),
            "asmc_protocol_payload": (
                asmc_protocol_for_id if asmc_config is not None else None
            ),
        },
        "compute": {
            "schema": "asmc-compute-v2",
            "C_int": "integrated attention positions with triangular prefill",
            "C_step": "sum of active batch sizes over model forward calls",
            "forward_calls": "literal number of model forward calls",
            "cap_semantics": (
                "checked after each forward-backed generation update; final "
                "C_int can overshoot by at most the last model forward"
            ),
        },
        "timing": {"schema": "synchronized-end-to-end-wall-clock-v1"},
        "rng": {
            "protocol": RNG_PROTOCOL,
            "derivation": (
                "uint32(first 8 hex digits of SHA-256 over canonical UTF-8 "
                "JSON containing base_seed, global problem_idx, method, "
                "method_identity, and rng_protocol)"
            ),
            "method_identities": {
                method: identity
                for method, identity in method_rng_identities.items()
                if rng_method_enabled[method]
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu_name,
            "nvidia_driver": nvidia_driver_version,
            "flash_attn": flash_attn_version,
        },
        "outputs": {
            "csv": os.path.basename(csv_path),
            "manifest": os.path.basename(manifest_path),
            "expected_rows": end - start,
            "completed_rows": 0,
        },
    }
    _write_manifest(manifest_path, manifest)
    
    # Print experiment info
    print(f"\n{'='*70}")
    only_asmc = args.run_asmc and not any(
        (
            args.run_greedy,
            args.run_naive,
            args.run_std,
            args.run_mcmc,
            args.run_majority,
            args.run_bestofn,
        )
    )
    experiment_label = "ASMC Experiment" if only_asmc else "Multi-method Experiment"
    print(f"{experiment_label}: {args.dataset}")
    print(f"{'='*70}")
    print(f"Model: {model_name} ({model_str})")
    print(f"Temperature: {args.temperature} (alpha = {args.alpha_star:.2f})")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Problems: {start} to {end-1}")
    print(f"\nMethods enabled:")
    print(f"  - Greedy: {args.run_greedy}")
    print(
        f"  - ASMC: {args.run_asmc} (N={args.n_particles}, "
        f"adaptive={args.enable_adaptive}, vote_mode={args.asmc_vote_mode}, "
        f"C_int cap={args.c_int_cap})"
    )
    print(f"  - Naive temp (1 sample): {args.run_naive}")
    print(f"  - Standard (1 sample): {args.run_std}")
    print(f"  - MCMC: {args.run_mcmc} (steps={args.mcmc_steps}, blocks={args.mcmc_blocks})")
    print(f"  - Majority Vote (N={args.n_particles} samples): {args.run_majority}")
    print(
        "  - Best-of-N: "
        f"{args.run_bestofn} (n={args.bestofn_n}, "
        f"temp={args.bestofn_temp}, chunk={args.bestofn_chunk_size})"
    )
    print(f"\nOutput: {csv_path}")
    print(f"{'='*70}\n")
    
    # Statistics
    stats = {
        'greedy': {'correct': 0, 'time': 0},
        'asmc': {'correct': 0, 'time': 0},
        'naive': {'correct': 0, 'time': 0},
        'std': {'correct': 0, 'time': 0},
        'mcmc': {'correct': 0, 'time': 0, 'accept': 0},
        'majority': {'correct': 0, 'time': 0},
        'bestofn': {'correct': 0, 'time': 0},
    }
    
    for problem_idx, data in enumerate(tqdm(dataset[start:end], desc="Comparison on MATH")):
        question = data["prompt"]
        answer = data["correct_answer"] if "correct_answer" in data else data["answer"]
        global_problem_idx = start + problem_idx
        
        if args.verbose:
            print(f"\n[Problem {global_problem_idx}] {question[:80]}...")
        
        # Format prompt
        input_text = format_prompt(question, model_name, tokenizer, args.cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(hf_model.device)
        context = [idx.item() for idx in input_ids[0]]
        
        result = {
            "problem_idx": global_problem_idx,
            "question": question,
            "correct_answer": answer,
            "run_id": timestamp,
            "model_id": model_str,
            "model_revision": model_revision,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "trust_remote_code": args.trust_remote_code,
            "dataset_name": dataset_name,
            "dataset_sha256": dataset_sha256,
            "code_git_commit": code_state["git_commit"],
            "code_git_dirty": code_state["dirty"],
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": gpu_name,
            "nvidia_driver_version": nvidia_driver_version,
            "flash_attn_version": flash_attn_version,
            "seed": args.seed,
            "rng_protocol": RNG_PROTOCOL,
            "batch_idx": args.batch_idx,
            "cot": args.cot,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "compute_schema": "asmc-compute-v2",
            "timing_schema": "synchronized-end-to-end-wall-clock-v1",
            **sampling_protocol_metadata,
            "asmc_mode": run_mode,
            "asmc_config": asmc_config_id if args.run_asmc else None,
            "asmc_protocol": (
                METHOD_PROTOCOLS["asmc"] if args.run_asmc else None
            ),
            "asmc_protocol_sha256": (
                asmc_protocol_sha256 if args.run_asmc else None
            ),
            "asmc_protocol_payload": (
                asmc_protocol_payload if args.run_asmc else None
            ),
            "asmc_backend": (
                "batched" if args.use_batched else "sequential"
            ) if args.run_asmc else None,
            "asmc_use_batched": args.use_batched if args.run_asmc else None,
            "asmc_vote_mode": args.asmc_vote_mode if args.run_asmc else None,
            "asmc_use_source_weight": (
                asmc_use_source_weight if args.run_asmc else None
            ),
            "asmc_c_int_cap": (
                args.c_int_cap if args.c_int_cap is not None else "none"
            ) if args.run_asmc else None,
            "asmc_legacy_stop_constraints": args.legacy_stop_constraints,
            "asmc_n_particles": (
                asmc_config.n_particles if asmc_config is not None else None
            ),
            "asmc_fast_n_particles": (
                asmc_config.fast_n_particles if asmc_config is not None else None
            ),
            "asmc_hard_n_particles": (
                asmc_config.hard_n_particles if asmc_config is not None else None
            ),
            "asmc_block_size": args.block_size,
            "asmc_ess_threshold": args.ess_threshold,
            "asmc_epsilon": args.epsilon,
            "asmc_alpha_start": args.alpha_start,
            "asmc_alpha_star": args.alpha_star,
            "asmc_anneal_tokens": args.anneal_tokens,
            "asmc_anneal_schedule": args.anneal_schedule,
            "asmc_early_stop_mass_threshold": (
                asmc_config.early_stop_mass_threshold
                if asmc_config is not None else None
            ),
            "asmc_early_stop_min_tokens": (
                asmc_config.early_stop_min_tokens
                if asmc_config is not None else None
            ),
            "asmc_early_stop_ess_frac": (
                asmc_config.early_stop_ess_frac
                if asmc_config is not None else None
            ),
            "asmc_early_stop_min_parsed_frac": (
                asmc_config.early_stop_min_parsed_frac
                if asmc_config is not None else None
            ),
            "asmc_early_stop_stable_checks": (
                asmc_config.early_stop_stable_checks
                if asmc_config is not None else None
            ),
            "asmc_fast_mass_threshold": (
                asmc_config.fast_mass_threshold
                if asmc_config is not None else None
            ),
            "asmc_hard_anneal_tokens": (
                asmc_config.hard_anneal_tokens
                if asmc_config is not None else None
            ),
            "asmc_hard_alpha_start": (
                asmc_config.hard_alpha_start
                if asmc_config is not None else None
            ),
            "asmc_hard_ess_threshold": (
                asmc_config.hard_ess_threshold
                if asmc_config is not None else None
            ),
            "asmc_hard_epsilon": 0.08 if asmc_config is not None else None,
            "asmc_hard_early_stop_mass_threshold": (
                asmc_config.hard_early_stop_mass_threshold
                if asmc_config is not None else None
            ),
            "asmc_hard_early_stop_min_tokens": (
                asmc_config.hard_early_stop_min_tokens
                if asmc_config is not None else None
            ),
            "asmc_hard_early_stop_ess_frac": (
                asmc_config.hard_early_stop_ess_frac
                if asmc_config is not None else None
            ),
            "asmc_hard_early_stop_min_parsed_frac": (
                asmc_config.hard_early_stop_min_parsed_frac
                if asmc_config is not None else None
            ),
            "greedy_mode": "single" if args.run_greedy else None,
            "greedy_config": "greedy" if args.run_greedy else None,
            "greedy_protocol": (
                METHOD_PROTOCOLS["greedy"] if args.run_greedy else None
            ),
            "greedy_pass_type": "single" if args.run_greedy else None,
            "naive_mode": "single" if args.run_naive else None,
            "naive_config": (
                f"temp{args.temperature:g}" if args.run_naive else None
            ),
            "naive_protocol": (
                METHOD_PROTOCOLS["naive"] if args.run_naive else None
            ),
            "naive_pass_type": "single" if args.run_naive else None,
            "std_mode": "single" if args.run_std else None,
            "std_config": "temp1" if args.run_std else None,
            "std_protocol": (
                METHOD_PROTOCOLS["std"] if args.run_std else None
            ),
            "std_pass_type": "single" if args.run_std else None,
            "mcmc_mode": "single" if args.run_mcmc else None,
            "mcmc_config": (
                f"steps{args.mcmc_steps}_blocks{args.mcmc_blocks}_"
                f"temp{args.temperature:g}"
                if args.run_mcmc
                else None
            ),
            "mcmc_protocol": (
                METHOD_PROTOCOLS["mcmc"] if args.run_mcmc else None
            ),
            "mcmc_steps": args.mcmc_steps if args.run_mcmc else None,
            "mcmc_blocks": args.mcmc_blocks if args.run_mcmc else None,
            "mcmc_temperature": (
                args.temperature if args.run_mcmc else None
            ),
            "mcmc_pass_type": "single" if args.run_mcmc else None,
            "majority_mode": "single" if args.run_majority else None,
            "majority_config": (
                f"n{args.n_particles}_temp{args.temperature:g}"
                if args.run_majority
                else None
            ),
            "majority_protocol": (
                METHOD_PROTOCOLS["majority"] if args.run_majority else None
            ),
            "majority_n": args.n_particles if args.run_majority else None,
            "majority_temperature": (
                args.temperature if args.run_majority else None
            ),
            "majority_pass_type": "single" if args.run_majority else None,
            "bestofn_mode": "single" if args.run_bestofn else None,
            "bestofn_config": bestofn_config_id if args.run_bestofn else None,
            "bestofn_protocol": (
                METHOD_PROTOCOLS["bestofn"] if args.run_bestofn else None
            ),
            "bestofn_pass_type": "single" if args.run_bestofn else None,
            "bestofn_n": args.bestofn_n if args.run_bestofn else None,
            "bestofn_temperature": (
                args.bestofn_temp if args.run_bestofn else None
            ),
            "bestofn_chunk_size": (
                args.bestofn_chunk_size if args.run_bestofn else None
            ),
        }

        # ============ 0. Greedy decoding ============
        if args.run_greedy:
            greedy_tracker = ComputeTracker()
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "greedy",
                    method_rng_identities["greedy"],
                )
                with instrument_model(hf_model, greedy_tracker):
                    greedy_ids, greedy_completion = greedy_sample(
                        autoreg_sampler, context, args.max_tokens
                    )
                greedy_compute = greedy_tracker.get_stats()
                greedy_answer = parse_answer_robust(greedy_completion)
                try:
                    is_correct = (
                        greedy_answer is not None
                        and grade_answer(greedy_answer, answer)
                    )
                except Exception:
                    is_correct = (
                        greedy_answer is not None
                        and str(greedy_answer).strip() == str(answer).strip()
                    )

                stats['greedy']['correct'] += int(is_correct)
                stats['greedy']['time'] += greedy_compute['total_time']
                result["greedy_completion"] = greedy_completion
                result["greedy_answer"] = greedy_answer
                result["greedy_correct"] = is_correct
                result["greedy_time"] = greedy_compute['total_time']
                _record_completion_evidence(
                    result,
                    "greedy",
                    greedy_ids,
                    tokenizer.eos_token_id,
                )
                result["greedy_prefill_flops"] = greedy_compute['prefill_flops']
                result["greedy_decode_flops"] = greedy_compute['decode_flops']
                result["greedy_total_flops"] = greedy_compute['total_flops']
                result["greedy_n_prefill"] = greedy_compute['n_prefill']
                result["greedy_n_decode"] = greedy_compute['n_decode']
                result["greedy_total_tokens"] = greedy_compute['total_tokens']
                _add_paper_compute_fields(result, "greedy", greedy_compute)
            except Exception as exc:
                result["greedy_completion"] = f"ERROR: {exc}"
                result["greedy_answer"] = None
                result["greedy_correct"] = False
                result["greedy_time"] = greedy_tracker.total_time
                result["greedy_prefill_flops"] = None
                result["greedy_decode_flops"] = None
                result["greedy_total_flops"] = None
                result["greedy_n_prefill"] = None
                result["greedy_n_decode"] = None
                result["greedy_total_tokens"] = None
                _record_completion_evidence(
                    result, "greedy", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "greedy", time_s=greedy_tracker.total_time
                )
        
        # ============ 1. ASMC ============
        if args.run_asmc:
            asmc_tracker = ComputeTracker()
            c = len(context)
            N = asmc_config.n_particles
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "asmc",
                    method_rng_identities["asmc"],
                )
                with instrument_model(hf_model, asmc_tracker):
                    particles, _, _, diagnostics = asmc_sampler.sample(
                        context,
                        asmc_config,
                        verbose=args.verbose,
                        tracker=asmc_tracker,
                    )
                asmc_compute = asmc_tracker.get_stats()

                # ===== RE-DO VOTING BASED ON args.asmc_vote_mode =====
                if args.asmc_vote_mode == "weighted":
                    best_answer, best_particle, vote_info = weighted_voting_output(
                        particles, tokenizer, c, asmc_config.alpha_star,
                        use_source_weight=asmc_config.use_source_weight)
                elif args.asmc_vote_mode == "weighted_no_source":
                    best_answer, best_particle, vote_info = weighted_voting_output(
                        particles, tokenizer, c, asmc_config.alpha_star,
                        use_source_weight=asmc_config.use_source_weight)
                elif args.asmc_vote_mode == "majority":
                    best_answer, best_particle, vote_info = unweighted_majority_voting(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=True)
                elif args.asmc_vote_mode == "majority_no_source":
                    best_answer, best_particle, vote_info = unweighted_majority_voting(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=False)
                else:
                    # Default fallback
                    best_answer, best_particle, vote_info = weighted_voting_output(
                        particles, tokenizer, c, asmc_config.alpha_star,
                        use_source_weight=asmc_config.use_source_weight)

                completion_ids = best_particle.tokens[c:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

                mass_top = vote_info.get("best_mass", 0.0)

                is_correct = False
                if best_answer is not None:
                    try:
                        is_correct = grade_answer(best_answer, answer)
                    except Exception:
                        is_correct = (str(best_answer).strip() == str(answer).strip())

                if is_correct:
                    stats['asmc']['correct'] += 1
                stats['asmc']['time'] += asmc_compute['total_time']

                result["asmc_completion"] = completion
                _record_completion_evidence(
                    result,
                    "asmc",
                    completion_ids,
                    tokenizer.eos_token_id,
                )
                result["asmc_answer"] = best_answer
                result["asmc_correct"] = is_correct
                result["asmc_time"] = asmc_compute['total_time']
                result["asmc_mass_top"] = mass_top
                result["asmc_n_resamples"] = diagnostics.get("n_resamples", 0)
                result["asmc_pass_type"] = diagnostics.get("pass_type", "single")
                result["asmc_vote_mode"] = args.asmc_vote_mode
                result["asmc_use_source_weight"] = asmc_config.use_source_weight
                result["asmc_budget_exhausted"] = diagnostics.get(
                    "budget_exhausted", False
                )
                result["asmc_budget_exhausted_at_token"] = diagnostics.get(
                    "budget_exhausted_at_token"
                )
                result["asmc_stop_reason"] = diagnostics.get("stop_reason")

                # ===== NEW DIAGNOSTIC FIELDS =====
                unique_anc_hist = diagnostics.get("unique_ancestors_history", [])
                result["asmc_unique_ancestors_min"] = min(unique_anc_hist) if unique_anc_hist else N
                result["asmc_unique_ancestors_avg"] = float(np.mean(unique_anc_hist)) if unique_anc_hist else float(N)
                result["asmc_unique_sequences_final"] = diagnostics.get("unique_sequences_final", N)
                result["asmc_n_parsed"] = vote_info.get("n_parsed", 0)
                result["asmc_parse_rate"] = (
                    vote_info.get("n_parsed", 0) / len(particles)
                    if particles else 0.0
                )

                # ===== COMPUTE TRACKING FIELDS =====
                result["asmc_prefill_flops"] = asmc_compute['prefill_flops']
                result["asmc_decode_flops"] = asmc_compute['decode_flops']
                result["asmc_total_flops"] = asmc_compute['total_flops']
                result["asmc_n_prefill"] = asmc_compute['n_prefill']
                result["asmc_n_decode"] = asmc_compute['n_decode']
                result["asmc_total_tokens"] = asmc_compute['total_tokens']
                _add_paper_compute_fields(result, "asmc", asmc_compute)

            except Exception as e:
                result["asmc_completion"] = f"ERROR: {e}"
                result["asmc_answer"] = None
                result["asmc_correct"] = False
                result["asmc_time"] = asmc_tracker.total_time
                result["asmc_mass_top"] = None
                result["asmc_n_resamples"] = None
                result["asmc_pass_type"] = "error"
                result["asmc_vote_mode"] = args.asmc_vote_mode
                result["asmc_use_source_weight"] = asmc_use_source_weight
                result["asmc_budget_exhausted"] = None
                result["asmc_budget_exhausted_at_token"] = None
                result["asmc_stop_reason"] = "error"
                result["asmc_unique_ancestors_min"] = None
                result["asmc_unique_ancestors_avg"] = None
                result["asmc_unique_sequences_final"] = None
                result["asmc_n_parsed"] = None
                result["asmc_parse_rate"] = None
                result["asmc_prefill_flops"] = None
                result["asmc_decode_flops"] = None
                result["asmc_total_flops"] = None
                result["asmc_n_prefill"] = None
                result["asmc_n_decode"] = None
                result["asmc_total_tokens"] = None
                _record_completion_evidence(
                    result, "asmc", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "asmc", time_s=asmc_tracker.total_time
                )
        
        # ============ 2. Naive Temperature Sampling ============
        if args.run_naive:
            naive_tracker = ComputeTracker()
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "naive",
                    method_rng_identities["naive"],
                )
                with instrument_model(hf_model, naive_tracker):
                    naive_ids, naive_completion = naive_temp_sample(
                        autoreg_sampler, context, args.temperature, args.max_tokens
                    )
                naive_compute = naive_tracker.get_stats()
                
                naive_answer = parse_answer_robust(naive_completion)
                
                is_correct = False
                if naive_answer is not None:
                    try:
                        is_correct = grade_answer(naive_answer, answer)
                    except Exception:
                        is_correct = (str(naive_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['naive']['correct'] += 1
                stats['naive']['time'] += naive_compute['total_time']
                
                result["naive_completion"] = naive_completion
                result["naive_answer"] = naive_answer
                result["naive_correct"] = is_correct
                result["naive_time"] = naive_compute['total_time']
                _record_completion_evidence(
                    result,
                    "naive",
                    naive_ids,
                    tokenizer.eos_token_id,
                )
                
                # ===== COMPUTE TRACKING FIELDS =====
                result["naive_prefill_flops"] = naive_compute['prefill_flops']
                result["naive_decode_flops"] = naive_compute['decode_flops']
                result["naive_total_flops"] = naive_compute['total_flops']
                result["naive_n_prefill"] = naive_compute['n_prefill']
                result["naive_n_decode"] = naive_compute['n_decode']
                result["naive_total_tokens"] = naive_compute['total_tokens']
                _add_paper_compute_fields(result, "naive", naive_compute)
                
            except Exception as e:
                result["naive_completion"] = f"ERROR: {e}"
                result["naive_answer"] = None
                result["naive_correct"] = False
                result["naive_time"] = naive_tracker.total_time
                result["naive_prefill_flops"] = None
                result["naive_decode_flops"] = None
                result["naive_total_flops"] = None
                result["naive_n_prefill"] = None
                result["naive_n_decode"] = None
                result["naive_total_tokens"] = None
                _record_completion_evidence(
                    result, "naive", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "naive", time_s=naive_tracker.total_time
                )
        
        # ============ 3. Standard Sampling (temp=1.0) ============
        if args.run_std:
            std_tracker = ComputeTracker()
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "std",
                    method_rng_identities["std"],
                )
                with instrument_model(hf_model, std_tracker):
                    std_ids, std_completion = std_sample(
                        autoreg_sampler, context, args.max_tokens
                    )
                std_compute = std_tracker.get_stats()
                
                std_answer = parse_answer_robust(std_completion)
                
                is_correct = False
                if std_answer is not None:
                    try:
                        is_correct = grade_answer(std_answer, answer)
                    except Exception:
                        is_correct = (str(std_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['std']['correct'] += 1
                stats['std']['time'] += std_compute['total_time']
                
                result["std_completion"] = std_completion
                result["std_answer"] = std_answer
                result["std_correct"] = is_correct
                result["std_time"] = std_compute['total_time']
                _record_completion_evidence(
                    result,
                    "std",
                    std_ids,
                    tokenizer.eos_token_id,
                )
                
                # ===== COMPUTE TRACKING FIELDS =====
                result["std_prefill_flops"] = std_compute['prefill_flops']
                result["std_decode_flops"] = std_compute['decode_flops']
                result["std_total_flops"] = std_compute['total_flops']
                result["std_n_prefill"] = std_compute['n_prefill']
                result["std_n_decode"] = std_compute['n_decode']
                result["std_total_tokens"] = std_compute['total_tokens']
                _add_paper_compute_fields(result, "std", std_compute)
                
            except Exception as e:
                result["std_completion"] = f"ERROR: {e}"
                result["std_answer"] = None
                result["std_correct"] = False
                result["std_time"] = std_tracker.total_time
                result["std_prefill_flops"] = None
                result["std_decode_flops"] = None
                result["std_total_flops"] = None
                result["std_n_prefill"] = None
                result["std_n_decode"] = None
                result["std_total_tokens"] = None
                _record_completion_evidence(
                    result, "std", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "std", time_s=std_tracker.total_time
                )
        
        # ============ 4. MCMC Power Sampling ============
        if args.run_mcmc:
            mcmc_tracker = ComputeTracker()
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "mcmc",
                    method_rng_identities["mcmc"],
                )
                with instrument_model(hf_model, mcmc_tracker):
                    mcmc_full_ids, mcmc_completion, accept_ratio = mcmc_power_sample(
                        autoreg_sampler, context, args.temperature, 
                        args.mcmc_steps, args.max_tokens, args.mcmc_blocks
                    )
                mcmc_ids = mcmc_full_ids[len(context):]
                mcmc_compute = mcmc_tracker.get_stats()
                
                mcmc_answer = parse_answer_robust(mcmc_completion)
                
                is_correct = False
                if mcmc_answer is not None:
                    try:
                        is_correct = grade_answer(mcmc_answer, answer)
                    except Exception:
                        is_correct = (str(mcmc_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['mcmc']['correct'] += 1
                stats['mcmc']['time'] += mcmc_compute['total_time']
                stats['mcmc']['accept'] += accept_ratio
                
                result["mcmc_completion"] = mcmc_completion
                result["mcmc_answer"] = mcmc_answer
                result["mcmc_correct"] = is_correct
                result["mcmc_time"] = mcmc_compute['total_time']
                result["mcmc_accept_ratio"] = accept_ratio
                _record_completion_evidence(
                    result,
                    "mcmc",
                    mcmc_ids,
                    tokenizer.eos_token_id,
                )
                
                # ===== COMPUTE TRACKING FIELDS =====
                result["mcmc_prefill_flops"] = mcmc_compute['prefill_flops']
                result["mcmc_decode_flops"] = mcmc_compute['decode_flops']
                result["mcmc_total_flops"] = mcmc_compute['total_flops']
                result["mcmc_n_prefill"] = mcmc_compute['n_prefill']
                result["mcmc_n_decode"] = mcmc_compute['n_decode']
                result["mcmc_total_tokens"] = mcmc_compute['total_tokens']
                _add_paper_compute_fields(result, "mcmc", mcmc_compute)
                
            except Exception as e:
                result["mcmc_completion"] = f"ERROR: {e}"
                result["mcmc_answer"] = None
                result["mcmc_correct"] = False
                result["mcmc_time"] = mcmc_tracker.total_time
                result["mcmc_accept_ratio"] = 0.0
                result["mcmc_prefill_flops"] = None
                result["mcmc_decode_flops"] = None
                result["mcmc_total_flops"] = None
                result["mcmc_n_prefill"] = None
                result["mcmc_n_decode"] = None
                result["mcmc_total_tokens"] = None
                _record_completion_evidence(
                    result, "mcmc", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "mcmc", time_s=mcmc_tracker.total_time
                )
        
        # ============ 5. Naive + Majority Voting (Fair Baseline) ============
        if args.run_majority:
            majority_tracker = ComputeTracker()
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "majority",
                    method_rng_identities["majority"],
                )
                with instrument_model(hf_model, majority_tracker):
                    maj_completion, maj_answer, vote_info = naive_majority_vote(
                        autoreg_sampler, context, args.temperature, 
                        args.max_tokens, n_samples=args.n_particles
                    )
                majority_compute = majority_tracker.get_stats()
                
                is_correct = False
                if maj_answer is not None:
                    try:
                        is_correct = grade_answer(maj_answer, answer)
                    except Exception:
                        is_correct = (str(maj_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['majority']['correct'] += 1
                stats['majority']['time'] += majority_compute['total_time']
                
                result["majority_completion"] = maj_completion
                result["majority_answer"] = maj_answer
                result["majority_correct"] = is_correct
                result["majority_time"] = majority_compute['total_time']
                result["majority_n_samples"] = vote_info["n_samples"]
                result["majority_n_valid"] = vote_info["n_valid"]
                result["majority_n_unique"] = vote_info["n_unique"]
                result["majority_best_count"] = vote_info["best_count"]
                result["majority_mass"] = vote_info["best_mass"]
                _record_completion_evidence(
                    result,
                    "majority",
                    vote_info["selected_token_ids"],
                    tokenizer.eos_token_id,
                )
                
                # ===== COMPUTE TRACKING FIELDS =====
                result["majority_prefill_flops"] = majority_compute['prefill_flops']
                result["majority_decode_flops"] = majority_compute['decode_flops']
                result["majority_total_flops"] = majority_compute['total_flops']
                result["majority_n_prefill"] = majority_compute['n_prefill']
                result["majority_n_decode"] = majority_compute['n_decode']
                result["majority_total_tokens"] = majority_compute['total_tokens']
                _add_paper_compute_fields(result, "majority", majority_compute)
                
            except Exception as e:
                result["majority_completion"] = f"ERROR: {e}"
                result["majority_answer"] = None
                result["majority_correct"] = False
                result["majority_time"] = majority_tracker.total_time
                result["majority_n_samples"] = args.n_particles
                result["majority_n_valid"] = 0
                result["majority_n_unique"] = 0
                result["majority_best_count"] = 0
                result["majority_mass"] = 0.0
                result["majority_prefill_flops"] = None
                result["majority_decode_flops"] = None
                result["majority_total_flops"] = None
                result["majority_n_prefill"] = None
                result["majority_n_decode"] = None
                result["majority_total_tokens"] = None
                _record_completion_evidence(
                    result, "majority", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "majority", time_s=majority_tracker.total_time
                )

        # ============ 6. Best-of-N sample-and-select baseline ============
        if args.run_bestofn:
            bestofn_tracker = ComputeTracker()
            try:
                _activate_method_rng(
                    result,
                    args.seed,
                    global_problem_idx,
                    "bestofn",
                    method_rng_identities["bestofn"],
                )
                bon = _load_baseline_support().sample_best_of_n(
                    hf_model,
                    tokenizer,
                    input_ids[0],
                    n=args.bestofn_n,
                    max_new_tokens=args.max_tokens,
                    temperature=args.bestofn_temp,
                    generation_batch_size=args.bestofn_chunk_size,
                    scoring_batch_size=args.bestofn_chunk_size,
                    length_normalize=True,
                    track_compute=True,
                    compute_tracker=bestofn_tracker,
                )
                if bon.compute is None:  # pragma: no cover - defensive guard
                    raise RuntimeError("Best-of-N compute instrumentation is missing")
                bestofn_compute = bon.compute
                bestofn_answer = parse_answer_robust(bon.completion)
                try:
                    is_correct = (
                        bestofn_answer is not None
                        and grade_answer(bestofn_answer, answer)
                    )
                except Exception:
                    is_correct = (
                        bestofn_answer is not None
                        and str(bestofn_answer).strip() == str(answer).strip()
                    )

                stats['bestofn']['correct'] += int(is_correct)
                stats['bestofn']['time'] += bestofn_compute['total_time']

                info = bon.to_info_dict()
                result["bestofn_completion"] = bon.completion
                result["bestofn_answer"] = bestofn_answer
                result["bestofn_correct"] = is_correct
                result["bestofn_time"] = bestofn_compute['total_time']
                _record_completion_evidence(
                    result,
                    "bestofn",
                    bon.candidates[bon.best_index].tokens,
                    tokenizer.eos_token_id,
                )
                result["bestofn_best_idx"] = info["best_idx"]
                result["bestofn_best_score"] = info["best_score"]
                result["bestofn_best_len"] = info["best_len"]
                result["bestofn_score_mean"] = info["score_mean"]
                result["bestofn_len_mean"] = info["len_mean"]
                result["bestofn_n_eos"] = sum(
                    candidate.has_eos for candidate in bon.candidates
                )
                result["bestofn_n_hit_limit"] = sum(
                    candidate.hit_limit for candidate in bon.candidates
                )
                result["bestofn_generation_chunk_initial"] = info[
                    "generation_chunk_initial"
                ]
                result["bestofn_generation_chunk_used"] = info[
                    "generation_chunk_used"
                ]
                result["bestofn_generation_oom_retries"] = info[
                    "generation_oom_retries"
                ]
                result["bestofn_score_chunk_initial"] = info[
                    "score_chunk_initial"
                ]
                result["bestofn_score_chunk_used"] = info[
                    "score_chunk_used"
                ]
                result["bestofn_score_oom_retries"] = info[
                    "score_oom_retries"
                ]
                result["bestofn_prefill_flops"] = bestofn_compute[
                    'prefill_flops'
                ]
                result["bestofn_decode_flops"] = bestofn_compute[
                    'decode_flops'
                ]
                result["bestofn_total_flops"] = bestofn_compute[
                    'total_flops'
                ]
                result["bestofn_n_prefill"] = bestofn_compute['n_prefill']
                result["bestofn_n_decode"] = bestofn_compute['n_decode']
                result["bestofn_total_tokens"] = bestofn_compute[
                    'total_tokens'
                ]
                _add_paper_compute_fields(
                    result, "bestofn", bestofn_compute
                )
            except Exception as exc:
                result["bestofn_completion"] = f"ERROR: {exc}"
                result["bestofn_answer"] = None
                result["bestofn_correct"] = False
                result["bestofn_time"] = bestofn_tracker.total_time
                result["bestofn_best_idx"] = None
                result["bestofn_best_score"] = None
                result["bestofn_best_len"] = None
                result["bestofn_score_mean"] = None
                result["bestofn_len_mean"] = None
                result["bestofn_n_eos"] = None
                result["bestofn_n_hit_limit"] = None
                result["bestofn_generation_chunk_initial"] = (
                    args.bestofn_chunk_size
                )
                result["bestofn_generation_chunk_used"] = None
                result["bestofn_generation_oom_retries"] = None
                result["bestofn_score_chunk_initial"] = (
                    args.bestofn_chunk_size
                )
                result["bestofn_score_chunk_used"] = None
                result["bestofn_score_oom_retries"] = None
                result["bestofn_prefill_flops"] = None
                result["bestofn_decode_flops"] = None
                result["bestofn_total_flops"] = None
                result["bestofn_n_prefill"] = None
                result["bestofn_n_decode"] = None
                result["bestofn_total_tokens"] = None
                _record_completion_evidence(
                    result, "bestofn", None, tokenizer.eos_token_id
                )
                _add_paper_compute_fields(
                    result, "bestofn", time_s=bestofn_tracker.total_time
                )
        
        results.append(result)
        
        # Save incrementally
        _write_results_csv(csv_path, results)
        manifest["outputs"]["completed_rows"] = len(results)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        
        # Progress report
        n_done = problem_idx + 1
        print(f"\n  [{n_done}/{end-start}] Progress:")
        if args.run_greedy:
            acc = stats['greedy']['correct'] / n_done * 100
            avg_t = stats['greedy']['time'] / n_done
            print(f"    Greedy:   {acc:.1f}% acc, {avg_t:.1f}s avg")
        if args.run_asmc:
            acc = stats['asmc']['correct'] / n_done * 100
            avg_t = stats['asmc']['time'] / n_done
            print(f"    ASMC:     {acc:.1f}% acc, {avg_t:.1f}s avg")
        if args.run_majority:
            acc = stats['majority']['correct'] / n_done * 100
            avg_t = stats['majority']['time'] / n_done
            print(f"    Majority: {acc:.1f}% acc, {avg_t:.1f}s avg (N={args.n_particles})")
        if args.run_bestofn:
            acc = stats['bestofn']['correct'] / n_done * 100
            avg_t = stats['bestofn']['time'] / n_done
            print(
                f"    Best-of-{args.bestofn_n}: {acc:.1f}% acc, "
                f"{avg_t:.1f}s avg"
            )
        if args.run_naive:
            acc = stats['naive']['correct'] / n_done * 100
            avg_t = stats['naive']['time'] / n_done
            print(f"    Naive:    {acc:.1f}% acc, {avg_t:.1f}s avg (1 sample)")
        if args.run_std:
            acc = stats['std']['correct'] / n_done * 100
            avg_t = stats['std']['time'] / n_done
            print(f"    Std:      {acc:.1f}% acc, {avg_t:.1f}s avg (1 sample)")
        if args.run_mcmc:
            acc = stats['mcmc']['correct'] / n_done * 100
            avg_t = stats['mcmc']['time'] / n_done
            avg_accept = stats['mcmc']['accept'] / n_done
            print(f"    MCMC:     {acc:.1f}% acc, {avg_t:.1f}s avg, {avg_accept:.2f} accept")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"{'='*70}")
    print(f"Total problems: {len(results)}")
    print("\nFinal Results (raw runs; use audited C_int summaries for compute matching):")
    print("  [Particle and multi-sample methods]")
    
    n = len(results)
    if args.run_asmc:
        acc = stats['asmc']['correct'] / n * 100
        avg_t = stats['asmc']['time'] / n
        print(f"  ASMC:     {stats['asmc']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    if args.run_majority:
        acc = stats['majority']['correct'] / n * 100
        avg_t = stats['majority']['time'] / n
        print(f"  Majority: {stats['majority']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    if args.run_bestofn:
        acc = stats['bestofn']['correct'] / n * 100
        avg_t = stats['bestofn']['time'] / n
        print(
            f"  Best-of-{args.bestofn_n}: "
            f"{stats['bestofn']['correct']}/{n} ({acc:.2f}%), "
            f"avg time: {avg_t:.2f}s"
        )
    
    print(f"\n  [Single sample baselines]")
    if args.run_greedy:
        acc = stats['greedy']['correct'] / n * 100
        avg_t = stats['greedy']['time'] / n
        print(f"  Greedy:   {stats['greedy']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    if args.run_naive:
        acc = stats['naive']['correct'] / n * 100
        avg_t = stats['naive']['time'] / n
        print(f"  Naive:    {stats['naive']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    if args.run_std:
        acc = stats['std']['correct'] / n * 100
        avg_t = stats['std']['time'] / n
        print(f"  Std:      {stats['std']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    if args.run_mcmc:
        acc = stats['mcmc']['correct'] / n * 100
        avg_t = stats['mcmc']['time'] / n
        avg_accept = stats['mcmc']['accept'] / n
        print(f"  MCMC:     {stats['mcmc']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s, accept: {avg_accept:.2f}")
    
    enabled_methods = {
        "greedy": args.run_greedy,
        "asmc": args.run_asmc,
        "naive": args.run_naive,
        "std": args.run_std,
        "mcmc": args.run_mcmc,
        "majority": args.run_majority,
        "bestofn": args.run_bestofn,
    }
    method_error_rows = {}
    for method, enabled in enabled_methods.items():
        if enabled:
            completion_column = f"{method}_completion"
            method_error_rows[method] = sum(
                str(row.get(completion_column, "")).startswith("ERROR:")
                for row in results
            )
    manifest["status"], exit_code = _completion_status(method_error_rows)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"]["completed_rows"] = len(results)
    manifest["outputs"]["method_error_rows"] = method_error_rows
    _write_manifest(manifest_path, manifest)

    print(f"\nResults saved to: {csv_path}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"{'='*70}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
