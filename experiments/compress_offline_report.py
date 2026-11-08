"""Losslessly compress a strict JSON report; preserve the source and refuse overwrite."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.replay_offline_report import parse_report_bytes  # noqa: E402


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compress_report(source: Path, output: Path, *, level: int = 6) -> dict:
    """Publish only verified complete gzip bytes; never delete the input.

    Validation reads the full JSON into memory. This is not a streaming JSON parser
    or a scientific replay. Gzip framing omits timestamps and filenames.
    """
    source, output = Path(source), Path(output)
    if type(level) is not int or not 0 <= level <= 9:
        raise ValueError("compression level must be an integer from 0 through 9")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists; refusing overwrite")
    raw = source.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raise ValueError("input is already gzip; use the original JSON report")
    parse_report_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with gzip.GzipFile(
                filename="", fileobj=stream, mode="wb", compresslevel=level, mtime=0
            ) as packed:
                packed.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        actual, restored_bytes = hashlib.sha256(), 0
        with gzip.open(temporary, "rb") as restored:
            while chunk := restored.read(1024 * 1024):
                actual.update(chunk)
                restored_bytes += len(chunk)
        if restored_bytes != len(raw) or actual.hexdigest() != expected:
            raise ValueError("gzip round-trip did not preserve the source bytes")
        compressed_hash = _digest_file(temporary)
        compressed_bytes = temporary.stat().st_size
        # Both paths share a directory/filesystem. Link creation atomically fails
        # if another process creates output after the initial existence check.
        os.link(temporary, output)
        return {
            "schema_version": 1,
            "operation": "lossless_gzip_compression",
            "source": str(source.resolve()),
            "output": str(output.resolve()),
            "source_bytes": len(raw),
            "gzip_bytes": compressed_bytes,
            "uncompressed_sha256": expected,
            "gzip_sha256": compressed_hash,
            "byte_round_trip_verified": True,
            "scientific_replay_performed": False,
            "source_deleted": False,
            "compression_level": level,
            "python": platform.python_version(),
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level", type=int, default=6)
    args = parser.parse_args()
    try:
        result = compress_report(args.report, args.out, level=args.level)
    except (ValueError, TypeError, OSError, EOFError, zlib.error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
