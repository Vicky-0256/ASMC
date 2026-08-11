# ASMC result artifacts

This directory is reserved for compact, reviewed outputs from the strict
MATH500 **ASMC-only** audit. Raw completions, scheduler logs, model caches, and
hand-edited tables do not belong here. Archive the five raw CSV shards and
their runner manifests in a versioned external store and record their hashes
in the reviewed release metadata.

The repository does not currently contain a corrected 500-problem GPU rerun.
No camera-ready ASMC accuracy, compute, or latency value is validated merely
because it appears in an old results directory.

## Release scope

The required artifact path is:

```text
five fixed-ASMC shards
  -> strict fixed audit
  -> fixed *.summary.json

five adaptive-ASMC shards
  -> strict adaptive audit
  -> adaptive *.summary.json
```

Baseline result auditing, compute-matched selection, table rendering, and the
full Appendix C are optional research workflows and are not ASMC release
requirements. The corresponding utilities and `.gitignore` exceptions remain
available for future comparison work, but an ASMC-only artifact does not need
to produce `compute_matched.*` or `table_compute_matched.*`.

## Historical ASMC results that must not be reused

- The historical fixed `N=16` and `N=64` aggregates combined an adaptive
  batch-0 shard with fixed shards for the other 400 problems.
- Historical adaptive top-answer mass multiplied normalized particle weights
  by parser-source reliability factors, rather than using the direct sum in
  the paper equation.
- The claimed adaptive per-instance `C_int` cap was not wired into the
  historical experiment runner.
- Historical compute artifacts did not consistently distinguish
  `C_step = sum_k B_k` from literal model-forward calls.
- The paper contains conflicting descriptions of end-to-end wall-clock and
  CUDA-event-only timing; the historical latency producer has not been
  established.
- The old ASMC values lack one clean, hash-linked raw-shard -> manifest ->
  audit-summary provenance chain.

The corrected code makes these choices explicit, but code changes do not
repair old measurements. See
[`../../docs/reproducibility.md`](../../docs/reproducibility.md) for the full
protocol and disclosures.

## Audit a fixed ASMC configuration

Run from the repository root after all five 100-problem shards have completed:

```bash
ASMC_CONFIG_ID='asmc-fixed-n64-weighted_no_source-REPLACE_WITH_RECORDED_HASH'
python analysis/result_audit.py \
  path/to/fixed/batch0.csv path/to/fixed/batch1.csv \
  path/to/fixed/batch2.csv path/to/fixed/batch3.csv \
  path/to/fixed/batch4.csv \
  --method asmc \
  --config "${ASMC_CONFIG_ID}" \
  --mode fixed \
  --require-provenance \
  --json-out results/paper/fixed_n64.summary.json
```

Replace the placeholder with the exact `asmc_config` value stored in the
CSV. The identifier is derived from the complete canonical ASMC protocol;
shortened human aliases such as `n64_ann1024` are intentionally rejected.

## Audit an adaptive ASMC configuration

Audit adaptive shards independently:

```bash
ASMC_CONFIG_ID='asmc-adaptive-n128-weighted_no_source-REPLACE_WITH_RECORDED_HASH'
python analysis/result_audit.py \
  path/to/adaptive/batch0.csv path/to/adaptive/batch1.csv \
  path/to/adaptive/batch2.csv path/to/adaptive/batch3.csv \
  path/to/adaptive/batch4.csv \
  --method asmc \
  --config "${ASMC_CONFIG_ID}" \
  --mode adaptive \
  --require-provenance \
  --json-out results/paper/adaptive_n128.summary.json
```

The adaptive run must have declared its exact per-instance `C_int` cap before
launch. The audit verifies the recorded protocol and cap semantics; it does
not infer a cap from an observed result or a camera-ready table value.
Its manifest must also distinguish `ASMC_ANNEAL_TOKENS` (the adaptive fast
pass) from `ASMC_HARD_ANNEAL_TOKENS` (the restarted hard pass), together with
`ASMC_HARD_ALPHA_START` and `ASMC_HARD_ESS_THRESHOLD`. The hard-pass epsilon
is not an override: the runner fixes it at `0.08` and the audit requires that
exact recorded value.

## Strict acceptance criteria

The default publication audit requires exactly 500 unique `problem_idx`
values `0..499` in the expected five non-overlapping batches. It also requires:

- fixed ASMC rows to record fixed/single-pass execution with no adaptive rows;
- adaptive ASMC rows to record only valid fast/hard decisions and the resolved
  fast/hard populations;
- adaptive rows to record the distinct fast and hard annealing durations,
  starting alphas, and ESS thresholds; the hard-pass epsilon must equal the
  runner's fixed, audited protocol constant `0.08`;
- one exact canonical ASMC configuration and protocol hash across all shards;
- direct normalized-weight voting by default, with no unlabelled historical
  parser-source weighting;
- an explicitly recorded adaptive cap, exhaustion state, and overshoot
  semantics where adaptive mode is released;
- one clean committed code state, immutable model revision, dataset checksum,
  dtype, generation cap, seed, and software/GPU environment;
- canonical compute and synchronized end-to-end timing schemas;
- exact question, gold answer, and batch-index agreement with the pinned
  MATH500 file;
- canonical completion token IDs and EOS evidence; and
- successful independent reconstruction of the ASMC RNG identity, robust
  answer parsing, grading, and correctness.

The audit computes accuracy, p50, p95, and mean `C_int` from the same validated
rows. A self-reported `asmc_correct`, config label, dataset hash string, or
seed is not sufficient evidence by itself.

The audit does not yet independently decode stored token IDs back to the raw
completion. Publication still requires an archived, hash-pinned tokenizer
snapshot and an offline decode-consistency check.

`--allow-legacy-aliases` is diagnostic only. Likewise,
`--expected-problems` may be used for a clearly labelled smoke or diagnostic
run, but never for a 500-problem release artifact.

## Files to commit

For the ASMC-only release, intentionally commit only compact reviewed files:

- this README;
- one `*.summary.json` per released fixed or adaptive configuration; and
- optionally one reviewed aggregate `summary.csv` containing only the ASMC
  summaries.

Keep raw CSVs and runner sidecar manifests in a versioned external archive;
publish their SHA-256 values with the release. Review every summary's input
paths/hashes, configuration, coverage, and aggregate metrics before commit.
Never choose shards by timestamp or silently replace a failed batch with a
historical file.

The following are permitted by the repository for optional future comparison
work but are not required here: `compute_matched.json`,
`compute_matched.csv`, `table_compute_matched.tex`,
`table_compute_matched.md`, and `table_compute_matched.manifest.json`.
