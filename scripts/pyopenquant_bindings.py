"""Static parser for the PyO3 bindings in ``crates/pyopenquant/src``.

Shared by ``generate_api_inventory.py`` (the docstring gate) and ``generate_python_stubs.py``
(the ``.pyi`` stubs). It reads the Rust source rather than the built extension so that both
gates run without compiling anything; ``python/tests/test_stubs.py`` and stubtest then
check the result against the real extension.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "crates" / "pyopenquant" / "src"

# Binding files whose functions may still lack a docstring. Each entry is being changed by a
# concurrent pull request, so its docstrings are left to a follow-up rather than written here
# and conflicted there. This list must only shrink: the gate fails if a listed module has
# become fully documented, so the entry is removed in the same change that documents it.
UNDOCUMENTED_ALLOWLIST: dict[str, str] = {
    "cla": "cla.rs is being rewritten for #168 (CLA dual-problem / Sharpe fix)",
    "filters": "filters.rs has in-flight binding changes",
    "sb_bagging": "sb_bagging.rs is being changed by #187",
}

_REGISTER_MODULE_RE = re.compile(r'PyModule::new\(\s*py\s*,\s*"([A-Za-z0-9_]+)"\s*\)')
_WRAP_RE = re.compile(r"wrap_pyfunction!\(\s*([A-Za-z0-9_]+)\s*,")
_PYFN_NAME_RE = re.compile(r'#\[pyfunction(?:\(\s*name\s*=\s*"([A-Za-z0-9_]+)"\s*\))?\]')
_TYPE_ALIAS_RE = re.compile(
    r"^(?:pub(?:\([a-z]+\))?\s+)?type\s+([A-Za-z0-9_]+)\s*=\s*(.*?);", re.S | re.M
)


@dataclass
class Param:
    name: str
    rust_type: str
    default: str | None = None  # Rust expression from #[pyo3(signature = ...)]


@dataclass
class Binding:
    module: str  # Python submodule name, e.g. "labeling"
    name: str  # Python function name
    rust_name: str
    file: Path
    line: int
    doc: str
    params: list[Param] = field(default_factory=list)
    ret: str = "()"
    keyword_only_from: int | None = None  # index of the first keyword-only parameter


def _split_top(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside of <>, (), [] and string literals."""
    out, depth, cur, in_str = [], 0, [], False
    prev = ""
    for ch in text:
        if in_str:
            cur.append(ch)
            if ch == '"' and prev != "\\":
                in_str = False
        elif ch == '"':
            in_str = True
            cur.append(ch)
        elif ch in "<([":
            depth += 1
            cur.append(ch)
        elif ch in ">)]":
            # `->` is not a closing bracket.
            if ch == ">" and prev == "-":
                cur.append(ch)
            else:
                depth -= 1
                cur.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        prev = ch
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _balanced(text: str, start: int, open_ch: str = "(", close_ch: str = ")") -> int:
    """Index just past the bracket that closes the one at ``text[start]``."""
    depth = 0
    in_str = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if ch == '"' and text[i - 1] != "\\":
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced brackets")


def _strip_line_comments(text: str) -> str:
    return "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines())


def type_aliases(path: Path) -> dict[str, str]:
    text = _strip_line_comments(path.read_text(encoding="utf-8"))
    return {m.group(1): " ".join(m.group(2).split()) for m in _TYPE_ALIAS_RE.finditer(text)}


def _doc_above(lines: list[str], idx: int) -> str:
    """The ``///`` block attached to the item whose first attribute is ``lines[idx]``.

    Walks upward over doc comments, plain comments and single-line attributes.
    """
    doc: list[str] = []
    i = idx - 1
    while i >= 0:
        s = lines[i].strip()
        if s.startswith("///"):
            body = s[3:]
            doc.append(body[1:] if body.startswith(" ") else body)
        elif s.startswith("//") or (s.startswith("#[") and s.endswith("]")):
            pass
        else:
            break
        i -= 1
    doc.reverse()
    return "\n".join(doc).strip("\n")


def parse_file(path: Path) -> list[Binding]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    module_m = _REGISTER_MODULE_RE.search(text)
    if not module_m:
        return []
    module = module_m.group(1)
    registered = set(_WRAP_RE.findall(text))
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    out: list[Binding] = []
    for idx, line in enumerate(lines):
        m = _PYFN_NAME_RE.match(line.strip())
        if not m:
            continue
        doc = _doc_above(lines, idx)
        # Attributes and doc lines between #[pyfunction] and `fn`.
        pos = offsets[idx + 1]
        sig_text: str | None = None
        while True:
            chunk = text[pos:].lstrip()
            pos = len(text) - len(chunk)
            if chunk.startswith("///"):
                nl = text.index("\n", pos)
                body = text[pos + 3 : nl]
                doc = (doc + "\n" if doc else "") + (body[1:] if body.startswith(" ") else body)
                pos = nl + 1
            elif chunk.startswith("//"):
                pos = text.index("\n", pos) + 1
            elif chunk.startswith("#["):
                end = _balanced(text, pos + 1, "[", "]")
                attr = text[pos:end]
                sm = re.match(r"#\[pyo3\(\s*signature\s*=\s*", attr)
                if sm:
                    open_at = pos + sm.end()
                    sig_text = text[open_at + 1 : _balanced(text, open_at) - 1]
                pos = end
            else:
                break
        fm = re.match(r"(?:pub(?:\([a-z]+\))?\s+)?fn\s+([A-Za-z0-9_]+)\s*(<[^(]*>)?\s*", text[pos:])
        if not fm:
            raise SystemExit(f"{path}:{idx + 1}: could not find the fn after #[pyfunction]")
        rust_name = fm.group(1)
        open_at = pos + fm.end()
        close_at = _balanced(text, open_at)
        params_text = _strip_line_comments(text[open_at + 1 : close_at - 1])
        rest = text[close_at:]
        rm = re.match(r"\s*->\s*(.*?)\s*(?:where\b[^{]*)?\{", rest, re.S)
        ret = " ".join(rm.group(1).split()) if rm else "()"
        params: list[Param] = []
        for p in _split_top(params_text):
            if not p:
                continue
            p = re.sub(r"#\[[^\]]*\]\s*", "", p)
            pname, ptype = p.split(":", 1)
            pname = pname.strip().removeprefix("mut ").strip()
            ptype = " ".join(ptype.split())
            if re.match(r"Python\s*<", ptype):
                continue
            params.append(Param(pname, ptype))
        binding = Binding(
            module=module,
            name=m.group(1) or rust_name,
            rust_name=rust_name,
            file=path,
            line=idx + 1,
            doc=doc,
            params=params,
            ret=ret,
        )
        if sig_text is not None:
            _apply_signature(binding, sig_text)
        if rust_name in registered:
            out.append(binding)
    return out


def _apply_signature(binding: Binding, sig_text: str) -> None:
    by_name = {p.name: p for p in binding.params}
    ordered: list[Param] = []
    for entry in _split_top(_strip_line_comments(sig_text)):
        if entry == "*":
            binding.keyword_only_from = len(ordered)
            continue
        if entry.startswith("*"):
            raise SystemExit(f"{binding.file}: *args/**kwargs are not supported by the stub parser")
        name, _, default = entry.partition("=")
        name = name.strip()
        if name not in by_name:
            raise SystemExit(f"{binding.file}: signature names unknown parameter {name!r}")
        param = by_name[name]
        param.default = default.strip() or None
        ordered.append(param)
    binding.params = ordered


def all_bindings() -> list[Binding]:
    out: list[Binding] = []
    for path in sorted(BINDINGS.glob("*.rs")):
        out.extend(parse_file(path))
    return out


def binding_file_stem(binding: Binding) -> str:
    return binding.file.stem


def undocumented(bindings: list[Binding]) -> list[Binding]:
    return [b for b in bindings if not b.doc.strip()]
