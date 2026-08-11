# Result Integrity

This document contains the detailed integrity boundary for the public ASMC
artifact. The root README intentionally keeps only a short disclosure so that
the method and its sequence-level power-sampling target remain discoverable.

## Current release boundary

The supported public artifact is the corrected fixed and adaptive ASMC path on
MATH500. The repository does not yet contain a corrected 500-problem GPU rerun,
so no historical camera-ready ASMC accuracy, compute, or latency number should
be described as reproduced.

## Historical protocol distinctions

1. Historical aggregates labelled as fixed-N ASMC mixed an adaptive batch-0
   shard with fixed shards for the other 400 problems. Those 500-row aggregates
   are protocol-contaminated and must not be reused.
2. `--legacy_stop_constraints` changes the effective decoding distribution. It
   is disabled by default and must not be pooled with unconstrained runs.
3. Historical answer voting multiplied particle weights by parser-source
   reliability. The public default follows the paper's normalized-weight
   equation (`weighted_no_source`); source weighting is a separately labelled
   legacy/ablation protocol.
4. The historical runner did not connect the paper's strict per-instance
   `C_int` cap. The public batched path records and enforces an explicit cap,
   with any final-forward overshoot exposed in the audit fields. The sequential
   reference rejects capped runs rather than returning a partially updated
   population.
5. The paper's `C_step` equation (`sum_k B_k`) and literal model
   `forward_calls` are different quantities. Both are reported and must not be
   interchanged.
6. The main evaluation describes synchronized end-to-end wall-clock latency,
   while the environment appendix describes CUDA-event kernel timing. The
   runner and the optional cache microbenchmark label these timing schemas
   separately.

## Evidence chain

An archival result requires five complete, disjoint ASMC shards, a clean
committed code state, immutable model and dataset identities, complete run
manifests, and a strict regenerated audit summary. The audit checks the pinned
MATH500 rows, stored completions, generated token evidence, protocol payload,
configuration identity, and method-local RNG keys. A hash-pinned tokenizer
snapshot plus offline decode-consistency check is still required before a
result can be called fully publication-certified.

Baseline-specific audits, compute-matched tables, Appendix C, HumanEval, GSM8K,
and historical figure reconstruction are outside the ASMC-only release gate.
Their utilities remain in the tree for research use but are not evidence for an
ASMC release.

For the commands, metadata schema, and five-shard workflow, see
[`reproducibility.md`](reproducibility.md). For compute and timing definitions,
see [`benchmarking.md`](benchmarking.md).
