# ASMC Standalone

Annealed Sequential Monte Carlo (ASMC) with KV Cache Optimization for Mathematical Reasoning.

## Directory Structure

```
asmc_standalone/
├── asmc_full_comparison.py   # Main experiment script
├── asmc_sampler.py           # Core ASMC algorithm
├── asmc_batched.py           # Batched GPU implementation with KV cache optimization
├── constants.py              # Prompt templates
├── answer_normalizer.py      # Answer parsing utilities
├── __init__.py
├── grader_utils/             # Answer grading utilities
│   ├── parse_utils.py
│   ├── math_grader.py
│   └── math_normalize.py
├── data/
│   └── MATH500.json          # MATH500 benchmark dataset
├── results/                  # Output directory
└── scripts/
    ├── run_full_comparison.sh  # SLURM script for all methods
    ├── run_asmc_only.sh        # SLURM script for ASMC only
    └── submit_all.sh           # Submit all 5 batches
```

## Quick Start

### Submit all batches (MATH500 = 5 batches × 100 problems)

```bash
cd asmc_standalone

# Full comparison (ASMC + MCMC + Naive + Standard) - ~24h per batch
./scripts/submit_all.sh

# ASMC only (faster) - ~6h per batch
./scripts/submit_all.sh asmc_only
```

### Run single batch locally

```bash
# Batch 0 (problems 0-99)
python asmc_full_comparison.py \
    --save_str=results/qwen_math \
    --model=qwen_math \
    --dataset=MATH \
    --cot \
    --batch_idx=0 \
    --seed=0 \
    --run_asmc \
    --use_batched \
    --enable_adaptive \
    --verbose
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_particles` | 64 | Number of particles (32 for fast pass) |
| `--alpha_start` | 1.5 | Initial inverse temperature |
| `--anneal_tokens` | 512 | Tokens to reach target alpha |
| `--ess_threshold` | 0.5 | ESS threshold for resampling |
| `--epsilon` | 0.05 | Defensive mixture weight |
| `--early_stop_mass` | 0.80 | Confidence threshold for early stop |
| `--max_tokens` | 3072 | Maximum generation length |
| `--temp` | 0.25 | Sampling temperature |

## Methods Comparison

| Method | Accuracy | Time/Problem | Description |
|--------|----------|--------------|-------------|
| **ASMC** | 77.8% | ~50s | Annealed SMC with KV cache optimization |
| MCMC | 72.8% | ~340s | Power sampling baseline |
| Naive | 70.0% | ~15s | Temperature sampling (T=0.25) |
| Standard | 52.4% | ~15s | Standard autoregressive sampling |

## Output Format

Results are saved as CSV files with columns:
- `problem_idx`: Problem index (0-499)
- `question`: Problem text
- `correct_answer`: Ground truth answer
- `asmc_answer`, `asmc_correct`, `asmc_time`: ASMC results
- `mcmc_answer`, `mcmc_correct`, `mcmc_time`: MCMC results
- `naive_answer`, `naive_correct`, `naive_time`: Naive results
- `std_answer`, `std_correct`, `std_time`: Standard results

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.49+ (for DynamicCache.reorder_cache)
- CUDA 11.4+
- ~40-60GB GPU memory (A100 80GB recommended)
