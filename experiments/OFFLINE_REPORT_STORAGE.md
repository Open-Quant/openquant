# Lossless offline report storage

Large synthetic reports can contain repeated weights and interval records. Gzip
reduces their disk footprint without dropping replications or changing numerical
values. This tool preserves the exact original UTF-8 JSON bytes, including whitespace.

```bash
python experiments/mean_robustness_lab.py --repetitions 200 --out /tmp/mean.json
python experiments/compress_offline_report.py /tmp/mean.json --out /tmp/mean.json.gz
python experiments/replay_offline_report.py /tmp/mean.json.gz --lab mean_robustness
```

Replay recognizes gzip by its header, not its filename, and applies the same strict
JSON rules after decompression: duplicate keys, nonfinite numbers and non-object
roots are rejected. Corrupt or truncated gzip input fails cleanly. The identity-only
mode also accepts compressed reports but does not execute the scientific experiment.

Compression checks JSON syntax, then verifies that decompression returns the same
byte count and SHA256 as the captured input. It does not certify source identity,
scientific contents or real-market applicability. Its output explicitly states that
no scientific replay was performed. Run replay separately for that comparison.

The original file is never deleted. Existing outputs, including symlinks, are not
overwritten. A verified complete temporary file is published with atomic hard-link
creation; a competing writer therefore cannot be overwritten. Failed writes leave
no published output and the temporary file is removed. Filesystems without hard-link
support fail rather than weakening this publication contract.

The gzip header omits timestamps and original filenames, making output reproducible
for the same input bytes, compression level, Python and zlib implementation. The
metadata records those versions and both hashes. Cross-version compressed-byte
identity is not promised; the decompressed scientific input remains the reference.

Validation still reads and parses the full decompressed document in memory. This is
not a streaming JSON parser or a memory limit. In one measured run, a 45,473,607-byte
covariance report compressed to 6,921,403 bytes in 1.51 seconds, with about 173 MiB
peak child-process RSS on this Linux environment. Those costs describe this file and
machine, not a general performance guarantee. Its scientific replay also matched.
