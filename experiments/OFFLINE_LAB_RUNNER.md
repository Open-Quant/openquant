# Configured offline lab runner

Run the existing synthetic research labs through one JSON configuration interface.
The runner uses the allowlist in `replay_offline_report.py`; it does not accept Python
expressions, import paths, commands, URLs, or arbitrary lab source files.

Use a fresh Python process in a trusted, unchanged checkout. Python 3.11+ and NumPy
are required. Some labs require SciPy; `list` identifies required and optional runtime
packages without importing the individual labs. The pinned research dependencies are
in [requirements-offline-research.txt](requirements-offline-research.txt).

## Inspect, configure, run

From the repository root, using an environment with the lab's dependencies:

```bash
python -m experiments.run_offline_lab list
python -m experiments.run_offline_lab defaults --lab point_in_time
```

`defaults` imports the selected lab and prints only its complete keyword defaults as
strict JSON. It never executes a study. Save that output to a **new** config file and
review it before running. Shell redirection can overwrite an existing file; that is
outside the runner's output protection.

For a small protocol smoke run, use the checked-in explicit configuration:

```bash
python -m experiments.run_offline_lab run \
  --lab point_in_time \
  --config experiments/configs/offline-smoke/point_in_time.json \
  --out /tmp/point-in-time-new-report.json
```

Choose an output that does not already exist. `python experiments/run_offline_lab.py`
supports the same subcommands. An absolute script path works from another directory;
module invocation above expects the repository root on Python's import path. Config
and output paths are interpreted relative to the current working directory.

`--config` is required. It must be a UTF-8 JSON object containing `run_lab` keyword
arguments, at most 1 MiB, with no duplicate keys, nonfinite values, or nesting deeper
than 64 containers (the root object counts as one). Unknown keywords and malformed
configs fail before study execution. Lab-specific validation still applies. Omitted
arguments retain the selected lab's existing defaults; an explicit `{}` intentionally
requests all those defaults. JSON values reach `run_lab` without type coercion.

**Review the actual workload.** Neither the config size limit nor the runner is a CPU,
memory, report-size, or elapsed-time budget. A short config can request a very expensive
study. The small fixtures in `configs/offline-smoke/` cover the interface, not the
statistical reliability or conclusions of full experiments.

## Report and receipt

The output is the lab's ordinary strict JSON report, including its effective config.
No scientific fields or calculations are changed. Five older `run_lab` functions omit
metadata that their own CLIs add: `selection_validation`, `preprocessing_leakage`,
`adaptive_sign_audit`, `cluster_sign`, and `covariance_estimation`. The runner supplies
those same provenance/performance fields when absent. Present but incorrect provenance
is not repaired, and unexpected missing provenance from other labs is rejected.

Before loading the lab, the runner hashes its registered source and dependency files.
It checks those files again after execution and verifies the report's source/runtime
identity before publication. This is a consistency check, not code signing, a sandbox,
or proof of authenticity. Keep the checkout unchanged throughout a run. An already
running Python session can have cached dependencies, so prefer the fresh-process CLI.

Stdout is a separate JSON receipt with the lab, absolute config/output paths, byte
counts, source/dependency hashes, runtime versions, and these explicit fields:

- `config_sha256`: SHA-256 of the exact input file bytes, including whitespace.
- `report_sha256`: SHA-256 of the exact report bytes written to disk.
- `identity_matched: true`: the source/runtime identity check passed.
- `scientific_replay_performed: false`: the runner did not execute a second study to
  verify the scientific results.

To perform that separate exact replay:

```bash
python -m experiments.replay_offline_report /tmp/point-in-time-new-report.json \
  --lab point_in_time
```

Replay compares scientific output exactly, excluding provenance and performance.
Floating-point/runtime differences can cause mismatches. Neither identity checks nor
matching synthetic results establish real-market profitability.

## File protection and failure boundaries

The runner never writes to the config. It refuses existing output files, directories,
hard links, and symlinks, including dangling symlinks, before loading the lab. It writes,
flushes, and fsyncs a same-directory temporary file, then publishes it by an atomic,
exclusive hard link. A competing creator's output is not overwritten. Filesystems that
do not support hard links fail without an overwrite fallback.

Failures before publication leave no new report; the runner attempts to remove only
its own temporary file. An interruption, receipt-output failure, or temporary-unlink
failure **after** successful publication can leave a complete report, and possibly the
temporary file. Inspect the destination before retrying. The runner does not promise
cleanup under arbitrary filesystem failure or crash durability, and it will not
silently overwrite an existing result on retry.

The small Python helpers `list_labs()`, `lab_defaults(lab)`, `read_config(path)`, and
`run_configured_lab(lab, config_path, output_path)` use the same contracts. Config and
output arguments are filesystem paths, not decoded dictionaries.
