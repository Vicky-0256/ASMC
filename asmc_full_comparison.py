#!/usr/bin/env python
"""
Full Comparison Experiment: ASMC vs Baselines on MATH Dataset

Compares:
1. ASMC (Annealed Sequential Monte Carlo)
2. Naive Temperature Sampling (temperature = 1/alpha)
3. Standard Sampling (temperature = 1.0, do_sample=True)
4. MCMC Power Sampling (autoregressive MCMC targeting p^alpha)

All methods target p^alpha where alpha = 1/temperature (default temp=0.25 -> alpha=4)
"""

import os
import sys
import json
import random
import argparse
import time
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import transformers

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_experiments'))

from grader_utils.parse_utils import parse_answer
from grader_utils.math_grader import grade_answer
from constants import PROMPT, COT, BASE


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
    """
    Naive temperature sampling: sample from p^(1/temp) = p^alpha
    Uses model.generate with temperature scaling.
    
    Matches original power_samp_math.py exactly:
    hf_model.generate(input_ids, max_new_tokens=3072, 
                      return_dict_in_generate=True, output_scores=True, temperature=temp)
    
    Returns: (tokens, completion_text)
    """
    device = sampler.device
    tokenizer = sampler.tokenizer
    model = sampler.model
    
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    
    # Match original power_samp_math.py exactly - NO do_sample
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temp,
    )
    
    generated_ids = output.sequences[0][len(context):].tolist()
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_ids, completion


@torch.no_grad()
def std_sample(sampler, context, max_new_tokens):
    """
    Standard sampling: sample from p (temperature=1.0, default)
    This is the baseline without any temperature scaling.
    
    Matches original: hf_model.generate(input_ids, max_new_tokens=3072, 
                      return_dict_in_generate=True, output_scores=True, do_sample=True)
    
    Returns: (tokens, completion_text)
    """
    device = sampler.device
    tokenizer = sampler.tokenizer
    model = sampler.model
    
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    
    # Match original exactly: no explicit temperature (defaults to 1.0)
    # No explicit eos_token_id/pad_token_id to match original
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    generated_ids = output.sequences[0][len(context):].tolist()
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_ids, completion


@torch.no_grad()
def naive_majority_vote(sampler, context, temp, max_new_tokens, n_samples=64):
    """
    Naive Temperature Sampling + Majority Voting (Fair baseline for ASMC)
    
    Generate N independent samples with temperature sampling, parse answers,
    and return the majority vote answer.
    
    This is a fair comparison to ASMC which uses N particles.
    
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
    answers = []
    
    # Generate N samples
    for _ in range(n_samples):
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,  # Fix: need this to access output.sequences
        )
        
        generated_ids = output.sequences[0][len(context):].tolist()
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)
        
        # Parse answer
        answer = parse_answer(completion)
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
    }
    
    return completions[best_idx], best_answer, vote_info


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
    
    # Ensure divisibility
    if max_new_tokens % block_num != 0:
        block_num = 1
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        
        prop = output.sequences[0].tolist()
        
        if len(output.logits) == 0:
            return prop, [], []
            
        unscaled_logits = torch.stack(output.logits, dim=0)
        scaled_logits = torch.stack(output.scores, dim=0)
        tokens = output.sequences[0][len(prefix):]
        
        if len(tokens) != unscaled_logits.shape[0]:
            # Handle mismatch
            return prop, [], []
        
        idx = tokens.view(-1, 1, 1)
        log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
        log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()
        
        return prop, log_probs_norm, log_probs_unnorm
    
    # Generate blocks with MCMC refinement
    for block_idx in range(block_num):
        target_len = len(gen) + jump_size
        gen, lp_norm, lp_unnorm = generate_with_temp(gen, target_len, temp)
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        
        # MCMC steps within block
        for _ in range(mcmc_steps):
            if len(gen) <= c:
                break
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)
            
            # Propose new continuation
            prop, log_prob_prop, target_log_prob_prop = generate_with_temp(gen[:idx], t, temp)
            s = len(prop)
            
            if len(log_prob_prop) == 0 or s <= idx:
                continue
                
            # Current log probs for comparison range
            end_idx = min(s - c, len(log_probs_norm))
            start_idx = idx - c
            
            if start_idx >= end_idx or start_idx < 0:
                continue
                
            log_prob_cur = log_probs_norm[start_idx:end_idx]
            target_log_prob_cur = log_probs_unnorm[start_idx:end_idx]
            
            # MH acceptance ratio
            log_r = (sum(target_log_prob_prop) + sum(log_prob_cur) 
                    - sum(target_log_prob_cur) - sum(log_prob_prop))
            
            # Accept with probability min(1, exp(log_r))
            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                # Update log probs
                if idx - c < len(log_probs_norm):
                    log_probs_norm[idx-c:] = log_prob_prop
                    log_probs_unnorm[idx-c:] = target_log_prob_prop
        
        # Early stop on EOS - truncate all arrays
        if tokenizer.eos_token_id in gen:
            eos_idx = gen.index(tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            # Match gold standard: truncate to eos_idx + 1 (NOT eos_idx + 1 - c)
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            break
    
    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    # Match power_samp_math.py: return full sequence (including context)
    # and decode full sequence (mcmc_completion includes prompt in gold standard)
    full_sequence = gen  # includes context
    completion = tokenizer.decode(full_sequence, skip_special_tokens=True)
    
    return full_sequence, completion, acceptance_ratio


def main():
    parser = argparse.ArgumentParser(description="Full Comparison: ASMC vs Baselines on MATH")
    
    # Basic settings
    parser.add_argument("--save_str", type=str, default="results/", help="Save directory")
    parser.add_argument("--model", type=str, default="qwen_math",
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--dataset", type=str, default="MATH", help="Dataset name")
    parser.add_argument("--cot", action="store_true", default=True, help="Use Chain-of-Thought")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--n_particles", type=int, default=64, help="Number of particles")
    parser.add_argument("--block_size", type=int, default=32, help="Block size for ESS check")
    parser.add_argument("--ess_threshold", type=float, default=0.5, help="ESS threshold for resampling")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Defensive mixture epsilon")
    parser.add_argument("--anneal_tokens", type=int, default=512, help="Annealing duration (tokens)")
    parser.add_argument("--alpha_start", type=float, default=1.5, help="Starting alpha")
    parser.add_argument("--anneal_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--early_stop_mass", type=float, default=0.80, help="Early stop mass threshold")
    parser.add_argument("--enable_adaptive", action="store_true", default=True, help="Enable adaptive budget")
    parser.add_argument("--fast_mass_threshold", type=float, default=0.65, help="Fast pass mass threshold")
    parser.add_argument("--hard_n_particles", type=int, default=96, help="Hard pass particle count")
    parser.add_argument("--use_batched", action="store_true", default=True, help="Use batched inference")
    
    # Method selection
    parser.add_argument("--run_asmc", action="store_true", default=True, help="Run ASMC")
    parser.add_argument("--run_naive", action="store_true", help="Run naive temp sampling (single sample)")
    parser.add_argument("--run_std", action="store_true", help="Run standard sampling (single sample)")
    parser.add_argument("--run_mcmc", action="store_true", help="Run MCMC power sampling")
    parser.add_argument("--run_majority", action="store_true", help="Run naive temp + majority voting (N samples)")

    # ASMC voting mode for ablation study
    parser.add_argument("--asmc_vote_mode", type=str, default="weighted",
        choices=["weighted", "weighted_no_source", "majority", "majority_no_source"],
        help="ASMC voting strategy: weighted (default), weighted_no_source, majority, majority_no_source")
    
    # Verbose
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set alpha_star from temperature if not specified
    if args.alpha_star is None:
        args.alpha_star = 1.0 / args.temperature
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    model_name = args.model
    device = args.device
    
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
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "MATH":
        json_file = os.path.join(
            os.path.dirname(__file__), 'data', 'MATH500.json'
        )
        dataset = json.load(open(json_file, "r"))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"Dataset loaded: {len(dataset)} problems")
    
    # Load model
    print(f"Loading model: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    print("Model loaded successfully")
    
    # Create autoregressive sampler for baselines
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, hf_model.device)
    
    # Import ASMC sampler
    asmc_sampler = None
    asmc_config = None
    if args.run_asmc:
        from asmc_sampler import ASMCConfig, ASMCSampler, weighted_voting_output, unweighted_majority_voting
        from asmc_batched import BatchedASMCSampler
        
        asmc_config = ASMCConfig(
            alpha_star=args.alpha_star,
            n_particles=args.n_particles,
            block_size=args.block_size,
            max_new_tokens=args.max_tokens,
            ess_threshold=args.ess_threshold,
            epsilon=args.epsilon,
            anneal_tokens=args.anneal_tokens,
            alpha_start=args.alpha_start,
            anneal_schedule=args.anneal_schedule,
            early_stop_mass_threshold=args.early_stop_mass,
            early_stop_min_tokens=64,
            enable_rejuvenation=False,
            enable_adaptive=args.enable_adaptive,
            fast_mass_threshold=args.fast_mass_threshold,
            hard_n_particles=args.hard_n_particles,
        )
        
        if args.use_batched:
            asmc_sampler = BatchedASMCSampler(hf_model, tokenizer, hf_model.device)
        else:
            from asmc_sampler import ASMCSampler
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        save_dir,
        f"full_comparison_temp{args.temperature}_batch{args.batch_idx}_seed{args.seed}_{timestamp}.csv"
    )
    
    # Print experiment info
    print(f"\n{'='*70}")
    print(f"Full Comparison Experiment: {args.dataset}")
    print(f"{'='*70}")
    print(f"Model: {model_name} ({model_str})")
    print(f"Temperature: {args.temperature} (alpha = {args.alpha_star:.2f})")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Problems: {start} to {end-1}")
    print(f"\nMethods enabled:")
    print(f"  - ASMC: {args.run_asmc} (N={args.n_particles}, adaptive={args.enable_adaptive}, vote_mode={args.asmc_vote_mode})")
    print(f"  - Naive temp (1 sample): {args.run_naive}")
    print(f"  - Standard (1 sample): {args.run_std}")
    print(f"  - MCMC: {args.run_mcmc} (steps={args.mcmc_steps}, blocks={args.mcmc_blocks})")
    print(f"  - Majority Vote (N={args.n_particles} samples): {args.run_majority}")
    print(f"\nOutput: {csv_path}")
    print(f"{'='*70}\n")
    
    # Statistics
    stats = {
        'asmc': {'correct': 0, 'time': 0},
        'naive': {'correct': 0, 'time': 0},
        'std': {'correct': 0, 'time': 0},
        'mcmc': {'correct': 0, 'time': 0, 'accept': 0},
        'majority': {'correct': 0, 'time': 0},
    }
    
    for problem_idx, data in enumerate(tqdm(dataset[start:end], desc="Comparison on MATH")):
        question = data["prompt"]
        answer = data["correct_answer"] if "correct_answer" in data else data["answer"]
        
        if args.verbose:
            print(f"\n[Problem {start + problem_idx}] {question[:80]}...")
        
        # Format prompt
        input_text = format_prompt(question, model_name, tokenizer, args.cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(hf_model.device)
        context = [idx.item() for idx in input_ids[0]]
        
        result = {
            "problem_idx": start + problem_idx,
            "question": question,
            "correct_answer": answer,
        }
        
        # ============ 1. ASMC ============
        if args.run_asmc:
            t0 = time.time()
            c = len(context)
            N = asmc_config.n_particles
            try:
                particles, _, _, diagnostics = asmc_sampler.sample(
                    context, asmc_config, verbose=args.verbose
                )
                elapsed = time.time() - t0

                # ===== RE-DO VOTING BASED ON args.asmc_vote_mode =====
                if args.asmc_vote_mode == "weighted":
                    best_answer, best_particle, vote_info = weighted_voting_output(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=True)
                elif args.asmc_vote_mode == "weighted_no_source":
                    best_answer, best_particle, vote_info = weighted_voting_output(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=False)
                elif args.asmc_vote_mode == "majority":
                    best_answer, best_particle, vote_info = unweighted_majority_voting(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=True)
                elif args.asmc_vote_mode == "majority_no_source":
                    best_answer, best_particle, vote_info = unweighted_majority_voting(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=False)
                else:
                    # Default fallback
                    best_answer, best_particle, vote_info = weighted_voting_output(
                        particles, tokenizer, c, asmc_config.alpha_star, use_source_weight=True)

                completion_ids = best_particle.tokens[c:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

                mass_top = vote_info.get("best_mass", 0.0)

                is_correct = False
                if best_answer is not None:
                    try:
                        is_correct = grade_answer(best_answer, answer)
                    except:
                        is_correct = (str(best_answer).strip() == str(answer).strip())

                if is_correct:
                    stats['asmc']['correct'] += 1
                stats['asmc']['time'] += elapsed

                result["asmc_completion"] = completion
                result["asmc_answer"] = best_answer
                result["asmc_correct"] = is_correct
                result["asmc_time"] = elapsed
                result["asmc_mass_top"] = mass_top
                result["asmc_n_resamples"] = diagnostics.get("n_resamples", 0)
                result["asmc_pass_type"] = diagnostics.get("pass_type", "single")
                result["asmc_vote_mode"] = args.asmc_vote_mode

                # ===== NEW DIAGNOSTIC FIELDS =====
                unique_anc_hist = diagnostics.get("unique_ancestors_history", [])
                result["asmc_unique_ancestors_min"] = min(unique_anc_hist) if unique_anc_hist else N
                result["asmc_unique_ancestors_avg"] = float(np.mean(unique_anc_hist)) if unique_anc_hist else float(N)
                result["asmc_unique_sequences_final"] = diagnostics.get("unique_sequences_final", N)
                result["asmc_n_parsed"] = vote_info.get("n_parsed", 0)
                result["asmc_parse_rate"] = vote_info.get("n_parsed", 0) / N

            except Exception as e:
                result["asmc_completion"] = f"ERROR: {e}"
                result["asmc_answer"] = None
                result["asmc_correct"] = False
                result["asmc_time"] = time.time() - t0
                result["asmc_mass_top"] = None
                result["asmc_n_resamples"] = None
                result["asmc_pass_type"] = "error"
                result["asmc_vote_mode"] = args.asmc_vote_mode
                result["asmc_unique_ancestors_min"] = None
                result["asmc_unique_ancestors_avg"] = None
                result["asmc_unique_sequences_final"] = None
                result["asmc_n_parsed"] = None
                result["asmc_parse_rate"] = None
        
        # ============ 2. Naive Temperature Sampling ============
        if args.run_naive:
            t0 = time.time()
            try:
                _, naive_completion = naive_temp_sample(
                    autoreg_sampler, context, args.temperature, args.max_tokens
                )
                elapsed = time.time() - t0
                
                naive_answer = parse_answer(naive_completion)
                
                is_correct = False
                if naive_answer is not None:
                    try:
                        is_correct = grade_answer(naive_answer, answer)
                    except:
                        is_correct = (str(naive_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['naive']['correct'] += 1
                stats['naive']['time'] += elapsed
                
                result["naive_completion"] = naive_completion
                result["naive_answer"] = naive_answer
                result["naive_correct"] = is_correct
                result["naive_time"] = elapsed
                
            except Exception as e:
                result["naive_completion"] = f"ERROR: {e}"
                result["naive_answer"] = None
                result["naive_correct"] = False
                result["naive_time"] = time.time() - t0
        
        # ============ 3. Standard Sampling (temp=1.0) ============
        if args.run_std:
            t0 = time.time()
            try:
                _, std_completion = std_sample(
                    autoreg_sampler, context, args.max_tokens
                )
                elapsed = time.time() - t0
                
                std_answer = parse_answer(std_completion)
                
                is_correct = False
                if std_answer is not None:
                    try:
                        is_correct = grade_answer(std_answer, answer)
                    except:
                        is_correct = (str(std_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['std']['correct'] += 1
                stats['std']['time'] += elapsed
                
                result["std_completion"] = std_completion
                result["std_answer"] = std_answer
                result["std_correct"] = is_correct
                result["std_time"] = elapsed
                
            except Exception as e:
                result["std_completion"] = f"ERROR: {e}"
                result["std_answer"] = None
                result["std_correct"] = False
                result["std_time"] = time.time() - t0
        
        # ============ 4. MCMC Power Sampling ============
        if args.run_mcmc:
            t0 = time.time()
            try:
                _, mcmc_completion, accept_ratio = mcmc_power_sample(
                    autoreg_sampler, context, args.temperature, 
                    args.mcmc_steps, args.max_tokens, args.mcmc_blocks
                )
                elapsed = time.time() - t0
                
                mcmc_answer = parse_answer(mcmc_completion)
                
                is_correct = False
                if mcmc_answer is not None:
                    try:
                        is_correct = grade_answer(mcmc_answer, answer)
                    except:
                        is_correct = (str(mcmc_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['mcmc']['correct'] += 1
                stats['mcmc']['time'] += elapsed
                stats['mcmc']['accept'] += accept_ratio
                
                result["mcmc_completion"] = mcmc_completion
                result["mcmc_answer"] = mcmc_answer
                result["mcmc_correct"] = is_correct
                result["mcmc_time"] = elapsed
                result["mcmc_accept_ratio"] = accept_ratio
                
            except Exception as e:
                result["mcmc_completion"] = f"ERROR: {e}"
                result["mcmc_answer"] = None
                result["mcmc_correct"] = False
                result["mcmc_time"] = time.time() - t0
                result["mcmc_accept_ratio"] = 0.0
        
        # ============ 5. Naive + Majority Voting (Fair Baseline) ============
        if args.run_majority:
            t0 = time.time()
            try:
                maj_completion, maj_answer, vote_info = naive_majority_vote(
                    autoreg_sampler, context, args.temperature, 
                    args.max_tokens, n_samples=args.n_particles
                )
                elapsed = time.time() - t0
                
                is_correct = False
                if maj_answer is not None:
                    try:
                        is_correct = grade_answer(maj_answer, answer)
                    except:
                        is_correct = (str(maj_answer).strip() == str(answer).strip())
                
                if is_correct:
                    stats['majority']['correct'] += 1
                stats['majority']['time'] += elapsed
                
                result["majority_completion"] = maj_completion
                result["majority_answer"] = maj_answer
                result["majority_correct"] = is_correct
                result["majority_time"] = elapsed
                result["majority_n_samples"] = vote_info["n_samples"]
                result["majority_n_valid"] = vote_info["n_valid"]
                result["majority_n_unique"] = vote_info["n_unique"]
                result["majority_best_count"] = vote_info["best_count"]
                result["majority_mass"] = vote_info["best_mass"]
                
            except Exception as e:
                result["majority_completion"] = f"ERROR: {e}"
                result["majority_answer"] = None
                result["majority_correct"] = False
                result["majority_time"] = time.time() - t0
                result["majority_n_samples"] = args.n_particles
                result["majority_n_valid"] = 0
                result["majority_n_unique"] = 0
                result["majority_best_count"] = 0
                result["majority_mass"] = 0.0
        
        results.append(result)
        
        # Save incrementally
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        # Progress report
        n_done = problem_idx + 1
        print(f"\n  [{n_done}/{end-start}] Progress:")
        if args.run_asmc:
            acc = stats['asmc']['correct'] / n_done * 100
            avg_t = stats['asmc']['time'] / n_done
            print(f"    ASMC:     {acc:.1f}% acc, {avg_t:.1f}s avg")
        if args.run_majority:
            acc = stats['majority']['correct'] / n_done * 100
            avg_t = stats['majority']['time'] / n_done
            print(f"    Majority: {acc:.1f}% acc, {avg_t:.1f}s avg (N={args.n_particles})")
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
    print(f"\nFinal Results (Fair Comparison):")
    print(f"  [Same compute budget: N={args.n_particles} samples]")
    
    n = len(results)
    if args.run_asmc:
        acc = stats['asmc']['correct'] / n * 100
        avg_t = stats['asmc']['time'] / n
        print(f"  ASMC:     {stats['asmc']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    if args.run_majority:
        acc = stats['majority']['correct'] / n * 100
        avg_t = stats['majority']['time'] / n
        print(f"  Majority: {stats['majority']['correct']}/{n} ({acc:.2f}%), avg time: {avg_t:.2f}s")
    
    print(f"\n  [Single sample baselines]")
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
    
    print(f"\nResults saved to: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
