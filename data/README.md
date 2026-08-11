# Data provenance

## Bundled data

`MATH500.json` contains 500 examples selected from the test split of the MATH benchmark. Each row stores the problem in `prompt`, the reference in `answer`, the source tag, and the original MATH-style path in `id`.

File integrity for the version currently bundled here:

```text
SHA-256  838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06  MATH500.json
Rows     500
```

Upstream benchmark:

- Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*
- Source repository: <https://github.com/hendrycks/math>

The current file does not encode the script, source revision, or sampling rule used to construct this particular 500-example subset. Before an archival release, the maintainers should either:

1. document the exact upstream dataset/revision and provide a deterministic construction script, or
2. identify the externally published MATH500 artifact from which this byte-identical file was obtained and record its revision and terms.

Do not assume that the eventual ASMC software license automatically applies to this dataset. Review and preserve the upstream dataset terms and attribution before redistribution.

## Evaluation helpers

Parts of `grader_utils/math_normalize.py` are identified in the source as being largely copied from the Hendrycks MATH evaluation release. The public release should preserve that provenance and any applicable upstream notice. The same principle applies to any evaluator imported from another benchmark.

## Data not bundled in the clean release

The paper also reports supplementary experiments on:

- HumanEval: <https://github.com/openai/human-eval>
- GSM8K: <https://github.com/openai/grade-school-math>

Obtain these datasets from their authoritative sources and record their revisions/checksums in each result manifest. Do not add downloaded copies to Git without first verifying their redistribution terms.

### HumanEval security warning

HumanEval executes model-generated Python. Treat every completion as untrusted code. Evaluation must run inside a disposable, network-isolated sandbox with no credentials or private data and with strict filesystem, process, time, and memory limits. A Python timeout or multiprocessing wrapper alone is not a security boundary.
