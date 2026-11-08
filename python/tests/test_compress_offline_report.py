"""Lossless storage, atomic publication and original-preservation contracts."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "compress_offline_report", ROOT / "experiments/compress_offline_report.py"
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_exact_bytes_and_deterministic_gzip(tmp_path):
    source = tmp_path / "report.json"
    raw = b'{\n  "synthetic_only": true, "unicode": "\\u03b1", "values": [1, 2.0]\n}\n'
    source.write_bytes(raw)
    first, second = tmp_path / "first.gz", tmp_path / "second.gz"
    a = lab.compress_report(source, first)
    b = lab.compress_report(source, second)
    assert gzip.decompress(first.read_bytes()) == raw == source.read_bytes()
    assert first.read_bytes() == second.read_bytes()
    assert a["uncompressed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert a["gzip_sha256"] == b["gzip_sha256"]
    assert not a["scientific_replay_performed"] and not a["source_deleted"]


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b"[]", b"garbage"])
def test_malformed_input_never_publishes(raw, tmp_path):
    source, output = tmp_path / "in.json", tmp_path / "out.gz"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        lab.compress_report(source, output)
    assert source.read_bytes() == raw and not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.json"]


def test_existing_output_and_aliases_preserved(tmp_path):
    source, output = tmp_path / "in.json", tmp_path / "out.gz"
    source.write_text("{}")
    output.write_bytes(b"existing")
    for target in (output, source):
        with pytest.raises(FileExistsError):
            lab.compress_report(source, target)
    assert output.read_bytes() == b"existing" and source.read_bytes() == b"{}"
    output.unlink()
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        lab.compress_report(source, output)
    assert output.is_symlink()


def test_concurrent_output_creation_is_not_overwritten(tmp_path, monkeypatch):
    source, output = tmp_path / "in.json", tmp_path / "out.gz"
    source.write_text("{}")
    real_link = lab.os.link

    def raced_link(temporary, destination):
        output.write_bytes(b"other writer")
        return real_link(temporary, destination)

    monkeypatch.setattr(lab.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        lab.compress_report(source, output)
    assert output.read_bytes() == b"other writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.json", "out.gz"]


def test_write_failure_removes_temporary_and_preserves_source(tmp_path, monkeypatch):
    source, output = tmp_path / "in.json", tmp_path / "out.gz"
    source.write_text("{}")

    def failed_sync(fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(lab.os, "fsync", failed_sync)
    with pytest.raises(OSError, match="disk failure"):
        lab.compress_report(source, output)
    assert source.read_bytes() == b"{}" and not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.json"]


def test_already_compressed_input_is_not_double_wrapped(tmp_path):
    source, output = tmp_path / "in.gz", tmp_path / "out.gz"
    source.write_bytes(gzip.compress(b"{}"))
    with pytest.raises(ValueError, match="already gzip"):
        lab.compress_report(source, output)
    assert not output.exists()


@pytest.mark.parametrize("level", [True, -1, 10, 1.5])
def test_invalid_level(level, tmp_path):
    source = tmp_path / "in.json"
    source.write_text(json.dumps({"x": 1}))
    with pytest.raises(ValueError):
        lab.compress_report(source, tmp_path / "out.gz", level=level)
