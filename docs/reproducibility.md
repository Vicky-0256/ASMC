# ASMC-only reproducibility and artifact-release guide

This guide defines the public artifact narrowly: run the corrected fixed and
adaptive cache-coherent ASMC protocols on MATH500 and verify their outputs with
the strict ASMC result audit.

## Reproducibility boundary

The supported path is:

```text
clean checkout
  -> ASMC-only smoke test
  -> five fixed or adaptive MATH500 shards
  -> analysis/result_audit.py --method asmc
  -> reviewed ASMC summary + raw-artifact hashes
```

The release is successful when a third party can:

- install the documented environment and load a pinned
  `Qwen/Qwen2.5-Math-7B` revision;
- run a one-problem cache-coherent ASMC smoke test;
- run exactly 500 MATH500 problems in five disjoint shards for a declared
  fixed configuration and a declared adaptive configuration; and
- regenerate one strict, content-addressed audit summary for each
  configuration from the raw shards.

Reproducing greedy, naive sampling, Best-of-N, MCMC, majority vote, a
compute-matched cross-method table, Appendix C, HumanEval, GSM8K, or every
paper figure is not part of this contract. Their code may remain in the tree
as unsupported research utilities, but their results and release readiness
do not block the ASMC artifact. In particular,
`analysis/select_compute_matched.py` and
`analysis/render_compute_table.py` are optional comparison utilities whose
strict path expects baseline inputs; do not use them as acceptance tests for
an ASMC-only release.

The current repository contains the corrected code and CPU tests, but it does
not contain a corrected 500-problem GPU rerun. No camera-ready ASMC accuracy,
compute, or latency value is therefore claimed as reproduced.

## Paper environment and unresolved pins

The paper describes the following primary setting:

| Component | Paper setting |
| --- | --- |
| Dataset | MATH500 |
| Model | `Qwen/Qwen2.5-Math-7B` |
| Precision | bfloat16 |
| Maximum new tokens | 3072 |
| Accelerator | NVIDIA A100-SXM-80GB |
| CUDA | PyTorch build with CUDA 12.x support |
| Main seed | 0 |

`requirements.txt` pins the Transformers version exercised by the offline
tiny-Qwen cache-reorder regression and gives compatibility ranges for the
remaining packages; it is not a bit-for-bit environment lock. The public
runner resolves `Qwen/Qwen2.5-Math-7B` to commit
`8daf1d676c3f24ddec5a99c5cff00a5c0e1c441c` when no revision is supplied.
This is the reproducible public-release snapshot, not a claim about the
unrecorded historical run. The corrected-run environment has not yet been
archived. Before publishing a result artifact, record at least:

```bash
python --version
python -m pip freeze
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
```

Also archive the exact tokenizer snapshot and its hash. The strict audit
validates the canonical token-ID evidence, vocabulary range, sequence length,
and EOS structure, but it does not yet independently decode those IDs and
compare them with the stored text. A hash-pinned tokenizer snapshot and an
offline decode-consistency check remain publication blockers.

## Stage 1: ASMC smoke test

Install the package requirements as described in the root README, then check
the entry point:

```bash
python asmc_full_comparison.py --help
```

Run one MATH500 problem with a deliberately small fixed population:

```bash
python asmc_full_comparison.py \
  --save_str results/smoke \
  --model qwen_math \
  --attn_implementation sdpa \
  --dataset MATH \
  --cot \
  --batch_idx 0 \
  --n_problems 1 \
  --seed 0 \
  --max_tokens 256 \
  --temp 0.25 \
  --n_particles 4 \
  --block_size 32 \
  --ess_threshold 0.5 \
  --epsilon 0.05 \
  --alpha_start 1.5 \
  --anneal_tokens 64 \
  --fixed \
  --run_asmc \
  --use_batched
```

This checks model loading, cache-coherent particle evolution, result writing,
and manifest writing. It is not a paper-quality evaluation and must not be
reported as one. The direct command selects PyTorch SDPA explicitly, matching
the `ASMC_RUN_PROFILE=smoke` script default, so this stage does not depend on
the optional `flash-attn` package. The script's full profile and the direct
runner default use FlashAttention 2 for the full GPU campaign.

## Stage 2: declare the ASMC campaign

Commit the complete ASMC sweep manifest before starting GPU jobs. At minimum,
declare separately for every fixed or adaptive configuration:

- mode, population sizes, block size, fast/fixed ESS threshold,
  defensive-mixture epsilon, proposal schedule, target
  temperature/exponent, vote mode, seed, and generation cap;
- adaptive hard-pass annealing duration, starting alpha, and ESS threshold;
- whether historical stop constraints are enabled;
- the adaptive top-answer-mass thresholds and an exact per-instance `C_int`
  cap, if adaptive mode is being reproduced;
- immutable code, model, tokenizer, and dataset identities; and
- expected hardware/software environment and the five batch indices.

Do not select a configuration or adaptive cap after inspecting its accuracy.
Unknown revisions or versions must remain unresolved rather than being
guessed.

The public default uses direct normalized-weight answer aggregation
(`weighted_no_source`) and disables the historical ASMC-only stop constraint.
This is the corrected, easier-to-interpret public protocol. Passing
`--legacy_stop_constraints` creates a distinct historical-protocol run; never
pool results produced with and without that flag.

Fixed and adaptive modes must be selected explicitly:

```text
--fixed
--adaptive
```

In adaptive mode, `--n_particles N` is the hard-pass population and the fast
pass defaults to `N/2`. `--hard_n_particles` should only differ from `N` for a
predeclared ablation. `--anneal_tokens`, `--alpha_start`,
`--ess_threshold`, and `--epsilon` configure the fast pass;
`--hard_anneal_tokens` (default `768`), `--hard_alpha_start` (default `1.3`),
and `--hard_ess_threshold` (default `0.6`) configure the restarted hard pass.
The hard-pass epsilon is a fixed protocol constant, `0.08`; it is recorded and
strictly audited rather than exposed as a CLI or environment override.
At a block boundary, resampling still occurs before answer-mass evaluation,
but the early-stop ESS guard uses the saved **pre-resampling** ESS. Resetting
particle weights during resampling therefore cannot make a collapsed
population pass the guard automatically.

## Stage 3: run all five MATH500 shards

The site-neutral SLURM wrapper runs ASMC without enabling baselines. Submit a
fixed configuration with:

```bash
ASMC_MODEL_REVISION=8daf1d676c3f24ddec5a99c5cff00a5c0e1c441c \
ASMC_MODE=fixed \
ASMC_RESULTS_DIR=results/runs/math500/fixed_n64 \
ASMC_N_PARTICLES=64 \
ASMC_ANNEAL_TOKENS=1024 \
./scripts/submit_all.sh asmc_only \
  --account=REPLACE_WITH_ACCOUNT --partition=REPLACE_WITH_GPU_PARTITION
```

Submit an adaptive configuration separately, with its cap chosen in the
committed campaign manifest:

```bash
ASMC_MODEL_REVISION=8daf1d676c3f24ddec5a99c5cff00a5c0e1c441c \
ASMC_MODE=adaptive \
ASMC_RESULTS_DIR=results/runs/math500/adaptive_n128 \
ASMC_N_PARTICLES=128 \
ASMC_HARD_N_PARTICLES=128 \
ASMC_ANNEAL_TOKENS=1536 \
ASMC_HARD_ANNEAL_TOKENS=768 \
ASMC_HARD_ALPHA_START=1.3 \
ASMC_HARD_ESS_THRESHOLD=0.6 \
ASMC_C_INT_CAP=REPLACE_WITH_NUMERIC_CAP \
./scripts/submit_all.sh asmc_only \
  --account=REPLACE_WITH_ACCOUNT --partition=REPLACE_WITH_GPU_PARTITION
```

These are explicit example configurations for exercising both public ASMC
modes; they are not a claim that the camera-ready hyperparameters or numerical
results have been reconstructed. Any released configuration must be declared
before execution, then run and audited. Supply site-specific account,
partition, environment, cache, and storage settings without editing the
recorded scientific configuration.

For non-SLURM systems, invoke `scripts/run_asmc_only.sh` once with each batch
index `0` through `4`, or translate that script to five equivalent direct
`asmc_full_comparison.py` invocations. Each batch covers 100 problems. Do not
use `--n_problems` for the full campaign.

## Stage 4: audit ASMC results

Run the audit from the repository root. A fixed configuration split across
five CSV files is audited as follows:

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

Use the exact `asmc_config` value stored in the CSV; it is derived from the
complete canonical ASMC protocol. Short aliases such as `n64_ann1024` are
intentionally rejected. Audit the adaptive shards independently:

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

The strict audit defaults to exactly 500 problems with IDs `0..499`. It:

- rejects missing, duplicate, overlapping, or wrong-batch rows;
- requires fixed rows to be single-pass and adaptive rows to record valid
  fast/hard decisions;
- verifies the question, gold answer, and dataset checksum against the pinned
  MATH500 bytes;
- re-parses every completion with the repository's robust parser and
  recomputes correctness;
- validates completion token IDs and EOS evidence;
- verifies the canonical ASMC protocol payload and configuration hash;
- independently reconstructs the isolated per-problem ASMC RNG key;
- checks the compute and timing schemas and ASMC diagnostics; and
- requires clean committed code, an immutable model revision, and complete
  software/GPU provenance when `--require-provenance` is used.

The audit reports accuracy, p50, p95, and mean `C_int` from the same validated
500 rows. It does not validate a historical camera-ready value merely because
that value appears elsewhere in the checkout. `--allow-legacy-aliases` and
`--expected-problems` overrides are diagnostic conveniences and must not be
used for a release artifact.

Cross-method selection and table rendering are optional and out of scope. An
ASMC-only release ends with reviewed fixed/adaptive audit summaries, their
manifests, and hashes of the externally archived raw shards.

## Historical ASMC integrity disclosures

### Mixed fixed/adaptive shards

Historical fixed `N=16` and `N=64` aggregates used an adaptive batch-0 shard
with fixed shards for the remaining 400 problems. Those 500-row aggregates are
protocol-contaminated and must not be reused. Fixed-only diagnostic
recalculations from legacy files are not corrected replacement results.

### Answer-mass weighting

The paper's top-answer-mass equation sums normalized particle weights for each
parsed answer directly. Historical code additionally multiplied them by
parser-source reliability weights. The corrected public default uses direct
weight aggregation. Source-weighted voting remains identifiable only as a
legacy/ablation protocol. Adaptive decisions affected by this difference need
a new GPU run.

### Adaptive compute cap

The paper claims a strict per-instance `C_int` cap, but the historical runner
did not connect that cap to ASMC execution. The corrected runner records the
cap and checks it after every successful forward-backed generation update.
Realized compute can exceed the requested cap by at most the final model
forward. This contract is implemented only by the cache-coherent batched
backend; the sequential reference rejects `--c_int_cap` because stopping its
per-particle loop midway would produce a partially updated population.
Exhaustion and overshoot are recorded. Historical adaptive rows must not be
called cap-enforced.

### Compute accounting

The paper's displayed definition is `C_step = sum_k B_k`, although nearby
prose describes a unit as one model call. The code follows the displayed
equation and stores literal model-forward invocations separately as
`forward_calls`/`*_n_forward`. `C_int`, `C_step`, and forward-call counts must
not be interchanged. The corrected implementation also distinguishes
quadratic prompt/prefix processing from cached multi-token decoding.

### Timing

The main evaluation text describes end-to-end wall-clock latency, while an
environment appendix describes CUDA-event kernel timing. The MATH500 runner
now records synchronized end-to-end wall clock. The optional cache-resampling
microbenchmark uses CUDA events and labels its schema separately. Historical
latencies cannot be reused until the producing implementation is identified.

### Provenance

The historical ASMC values were not derived from one authoritative,
hash-linked raw-shard and manifest chain. A corrected result must come from
five disjoint shards produced by one clean committed protocol and must be
regenerated through the strict ASMC audit. Code repairs are not retroactive
validation: no corrected GPU rerun is currently included.

## Required run manifest

Each logical ASMC configuration should have one machine-readable manifest. At
minimum, record:

```yaml
schema_version: 1
code:
  git_commit: null
  dirty: false
model:
  id: Qwen/Qwen2.5-Math-7B
  revision: 8daf1d676c3f24ddec5a99c5cff00a5c0e1c441c
  tokenizer_sha256: null
  dtype: bfloat16
data:
  name: MATH500
  path: data/MATH500.json
  sha256: 838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06
protocol:
  mode: fixed  # or adaptive
  backend: batched
  cot: true
  legacy_stop_constraints: false
  vote_mode: weighted_no_source
  use_source_weight: false
  c_int_cap: null  # predeclare a numeric value for adaptive mode
  seed: 0
  max_new_tokens: 3072
  temperature: 0.25
  n_particles: null
  hard_n_particles: null
  block_size: 32
  ess_threshold: 0.5
  epsilon: 0.05
  alpha_start: 1.5
  anneal_tokens: null  # fixed pass, or adaptive fast pass
  hard_anneal_tokens: 768
  hard_alpha_start: 1.3
  hard_ess_threshold: 0.6
  hard_epsilon: 0.08  # fixed runner constant; not user-configurable
  anneal_schedule: cosine
  early_stop_mass_threshold: 0.80
  fast_mass_threshold: 0.65
compute:
  schema: asmc-compute-v2
  cap_semantics: checked-after-forward
timing:
  schema: synchronized-end-to-end-wall-clock-v1
environment:
  python: null
  pytorch: null
  transformers: 4.46.3
  flash_attention: null
  cuda_runtime: null
  driver: null
  gpu: NVIDIA A100-SXM-80GB
results:
  expected_rows: 500
  raw_artifacts: null
  raw_sha256: null
  audit_summary: null
  audit_summary_sha256: null
```

The runner fills observable fields automatically. A dirty commit identifier is
useful provenance, but is not evidence that the commit alone reproduces the
run. Do not replace unknown revisions or versions with guesses.

## Determinism and raw artifacts

Record Python, NumPy, CPU-Torch, and CUDA seeds. Exact bitwise equality is not
guaranteed across GPU architectures, CUDA libraries, FlashAttention kernels,
or model revisions. The per-problem ASMC RNG identity prevents unrelated
method selection from perturbing ASMC's random stream, but it does not remove
those platform dependencies.

Raw completions can be large. Store the five raw CSVs and neighboring runner
manifests in a versioned external archive, and commit compact audit summaries,
configuration manifests, and all input hashes. Never construct a result by
scanning an uncontrolled results directory or choosing the latest filename.

## ASMC-only pre-release checklist

- [ ] Select and add a repository-level software license; preserve applicable
  third-party notices separately.
- [ ] Resolve the exact MATH500 subset provenance and redistribution terms.
- [ ] Freeze a clean canonical code commit for the corrected ASMC protocol.
- [ ] Commit the complete reviewed fixed/adaptive ASMC sweep manifest before
  launching jobs.
- [ ] Confirm the recorded immutable model revision and archive the tested
  environment lock.
- [ ] Archive and hash the tokenizer snapshot; add offline token-ID-to-text
  decode-consistency validation.
- [ ] Run the one-problem smoke command from a clean checkout.
- [ ] Re-run each released fixed configuration on all 500 problems, including
  configurations affected by the adaptive batch-0 mix-up.
- [ ] Re-run each released adaptive configuration with direct answer-mass
  aggregation and an explicitly recorded per-instance cap.
- [ ] Strictly audit the five fixed shards and five adaptive shards with
  `analysis/result_audit.py --method asmc`.
- [ ] Publish reviewed ASMC manifests/summaries and externally archive the raw
  shards with hashes.
- [ ] Add a GPU test comparing KV-reordered continuation logits with prefix
  replay.
- [ ] Verify that no cluster paths, user names, caches, secrets, scheduler
  logs, or oversized artifacts are tracked.
- [ ] Tag and archive the audited ASMC-only release.

Baseline reruns, comparison-table selection/rendering, full Appendix C,
HumanEval, GSM8K, and unrelated paper figures are intentionally absent from
this checklist.
