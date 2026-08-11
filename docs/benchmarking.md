# Benchmarking and Compute Accounting

The public runner is designed to make ASMC compute and latency semantics
explicit. This page records the definitions; it does not publish a new result
table.

## Compute fields

- `C_int` is the integrated-attention compute estimate used for ASMC budget
  matching and adaptive caps.
- `C_step` follows the paper's literal block equation, `sum_k B_k`.
- `forward_calls` counts successful model forwards separately from `C_step`.
- A capped adaptive run must use the cache-coherent batched backend. The final
  forward may produce a small auditable overshoot because the cap is checked
  after a successful forward-backed update.

The full CSV schema and strict audit command are documented in
[`reproducibility.md`](reproducibility.md) and
[`results/paper/README.md`](../results/paper/README.md).

## Latency

The MATH500 runner reports synchronized end-to-end wall-clock time, including
Python dispatch and cache-management overhead. The optional
[`microbench/cache_resampling.py`](../microbench/cache_resampling.py) benchmark
uses CUDA events for isolated GPU timing and labels that schema separately.
Measurements depend on GPU model, CUDA, PyTorch, Transformers, FlashAttention,
and runtime settings; compare only runs with matching manifests.

## Cache-reorder comparison

The microbenchmark compares ancestor-prefix replay with KV-cache gather plus a
single decode step. It is a systems diagnostic for cache coherence, not a
replacement for the ASMC MATH500 result audit.

## Baselines

Greedy, naive sampling, Best-of-N, MCMC, and majority-vote code remains as
optional research utilities. Baseline reproduction and a compute-matched
cross-method table are outside the ASMC-only public release contract.
