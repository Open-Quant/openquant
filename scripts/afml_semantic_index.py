#!/usr/bin/env python3
"""Build and query a semantic index for AFML PDF content.

Features:
- Extracts PDF text page-by-page with PyMuPDF
- Removes repeated header/footer boilerplate lines
- Performs hierarchy-aware segmentation (part/chapter/section/snippet)
- Chunks section bodies with Chonkie (RecursiveChunker)
- Embeds chunks with sentence-transformers
- Stores local artifacts for semantic retrieval
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from chonkie import RecursiveChunker
from sentence_transformers import SentenceTransformer

RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
RE_SOFT_LINEBREAK = re.compile(r"(?<!\n)\n(?!\n)")
RE_PAGE_ONLY = re.compile(r"^\s*\d+\s*$")

RE_PART = re.compile(r"^(PART\s+[IVXLC0-9]+)\s*(.*)$", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^(CHAPTER\s+\d+)\s*(.*)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^(\d+(?:\.\d+){1,3})\s+(.+)$")
RE_SNIPPET = re.compile(r"^(SNIPPET\s+\d+(?:\.\d+)*)\s+(.+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFML PDF semantic index builder/search")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build semantic index from PDF")
    build.add_argument("--pdf-path", required=True)
    build.add_argument("--out-dir", default="afml/doc_1")
    build.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    build.add_argument("--chunk-size", type=int, default=2200, help="Character-based chunk size")
    build.add_argument("--chunk-overlap", type=int, default=220, help="Character overlap between chunks")
    build.add_argument("--min-chars", type=int, default=180, help="Minimum chunk length to keep")
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument(
        "--allow-online-model-fetch",
        action="store_true",
        help="Allow Hugging Face network calls when loading embedding model",
    )

    search = sub.add_parser("search", help="Query built semantic index")
    search.add_argument("--index-dir", default="afml/doc_1/index")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--semantic-weight", type=float, default=0.86)
    search.add_argument("--chapter", default="", help="Optional chapter filter, e.g. 'CHAPTER 3'")
    search.add_argument("--json-output", action="store_true", help="Print machine-readable JSON results")
    search.add_argument(
        "--allow-online-model-fetch",
        action="store_true",
        help="Allow Hugging Face network calls when loading embedding model",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_HYPHEN_BREAK.sub(r"\1\2", text)
    text = RE_SOFT_LINEBREAK.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NL.sub("\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return RE_MULTI_SPACE.sub(" ", line.strip())


def is_probably_boilerplate(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if RE_PAGE_ONLY.match(s):
        return True
    if len(s) < 3:
        return True
    return False


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text") or ""
        pages.append({"page": i, "raw_text": raw})
    doc.close()
    return pages


def detect_repeated_margin_lines(pages: list[dict[str, Any]]) -> set[str]:
    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for p in pages:
        lines = [normalize_line(ln) for ln in p["raw_text"].splitlines() if ln.strip()]
        if not lines:
            continue

        top_lines = [ln for ln in lines[:3] if not is_probably_boilerplate(ln)]
        bottom_lines = [ln for ln in lines[-3:] if not is_probably_boilerplate(ln)]
        top_counter.update(top_lines)
        bottom_counter.update(bottom_lines)

    threshold = max(3, math.ceil(len(pages) * 0.08))
    repeated = {line for line, c in top_counter.items() if c >= threshold}
    repeated |= {line for line, c in bottom_counter.items() if c >= threshold}
    return repeated


def clean_pages(pages: list[dict[str, Any]], repeated_margin_lines: set[str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for p in pages:
        kept_lines: list[str] = []
        for line in p["raw_text"].splitlines():
            s = normalize_line(line)
            if not s:
                kept_lines.append("")
                continue
            if s in repeated_margin_lines:
                continue
            if RE_PAGE_ONLY.match(s):
                continue
            kept_lines.append(s)

        text = normalize_text("\n".join(kept_lines))
        cleaned.append({"page": p["page"], "text": text, "lines": kept_lines})
    return cleaned


def classify_heading(line: str) -> dict[str, str] | None:
    s = normalize_line(line)
    if not s:
        return None

    m = RE_PART.match(s)
    if m:
        return {"level": "part", "key": m.group(1).upper(), "title": m.group(1).upper()}

    m = RE_CHAPTER.match(s)
    if m:
        # Canonical chapter key (e.g., "CHAPTER 3") for stable filtering.
        return {"level": "chapter", "key": m.group(1).upper(), "title": m.group(1).upper()}

    m = RE_SECTION.match(s)
    if m:
        return {"level": "section", "key": m.group(1), "title": s}

    m = RE_SNIPPET.match(s)
    if m:
        return {"level": "snippet", "key": m.group(1).upper(), "title": s}

    if len(s) <= 90 and s.upper() == s and any(c.isalpha() for c in s):
        return {"level": "heading", "key": s, "title": s}

    return None


def build_sections(cleaned_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    part_title = ""
    chapter_title = ""
    section_title = ""

    curr_lines: list[str] = []
    curr_start_page: int | None = None
    curr_end_page: int | None = None

    def flush_section() -> None:
        nonlocal curr_lines, curr_start_page, curr_end_page
        text = normalize_text("\n".join(curr_lines))
        if not text:
            curr_lines = []
            curr_start_page = None
            curr_end_page = None
            return

        sec_idx = len(sections) + 1
        sections.append(
            {
                "section_id": f"S{sec_idx:05d}",
                "part": part_title,
                "chapter": chapter_title,
                "section": section_title,
                "page_start": curr_start_page,
                "page_end": curr_end_page,
                "text": text,
            }
        )
        curr_lines = []
        curr_start_page = None
        curr_end_page = None

    for p in cleaned_pages:
        page_num = int(p["page"])
        for line in p["lines"]:
            s = normalize_line(line)
            if not s:
                curr_lines.append("")
                continue

            heading = classify_heading(s)
            if heading is not None:
                # Start a new semantic section when encountering structural headings.
                flush_section()
                if heading["level"] == "part":
                    part_title = heading["title"]
                    chapter_title = ""
                    section_title = ""
                elif heading["level"] == "chapter":
                    chapter_title = heading["title"]
                    section_title = ""
                elif heading["level"] in {"section", "snippet", "heading"}:
                    section_title = heading["title"]

                curr_start_page = page_num
                curr_end_page = page_num
                curr_lines = [heading["title"]]
                continue

            if curr_start_page is None:
                curr_start_page = page_num
            curr_end_page = page_num
            curr_lines.append(s)

    flush_section()
    return sections


def chapter_list_from_sections(sections: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in sections:
        chapter = (s.get("chapter") or "").strip()
        if chapter and chapter not in seen:
            seen.add(chapter)
            out.append(chapter)
    return out


def build_chunks(
    sections: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    chunker = RecursiveChunker(chunk_size=chunk_size)

    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    for section in sections:
        text = section["text"]
        if len(text) < min_chars:
            continue

        sec_chunks = chunker.chunk(text)
        if not sec_chunks:
            continue

        section_out: list[dict[str, Any]] = []
        for c in sec_chunks:
            ctext = (c.text or "").strip()
            if len(ctext) < min_chars:
                continue

            heading_path = " > ".join(
                [
                    x
                    for x in [
                        (section.get("part") or "").strip(),
                        (section.get("chapter") or "").strip(),
                        (section.get("section") or "").strip(),
                    ]
                    if x
                ]
            )

            section_out.append(
                {
                    "chunk_id": f"afml-{section['section_id']}-c{chunk_idx:06d}",
                    "section_id": section["section_id"],
                    "page_start": int(section.get("page_start") or -1),
                    "page_end": int(section.get("page_end") or -1),
                    "page": int(section.get("page_start") or -1),
                    "part": section.get("part") or "",
                    "chapter": section.get("chapter") or "",
                    "section": section.get("section") or "",
                    "heading_path": heading_path,
                    "text": ctext,
                    "char_len": len(ctext),
                    "token_count": int(getattr(c, "token_count", 0) or 0),
                    "start_index": int(getattr(c, "start_index", -1) or -1),
                    "end_index": int(getattr(c, "end_index", -1) or -1),
                }
            )
            chunk_idx += 1

        if chunk_overlap > 0 and len(section_out) > 1:
            overlapped: list[dict[str, Any]] = []
            for i, ch in enumerate(section_out):
                if i == 0:
                    overlapped.append(ch)
                    continue
                prev = section_out[i - 1]
                tail = prev["text"][-chunk_overlap:]
                merged = (tail + "\n\n" + ch["text"]).strip()
                ch2 = dict(ch)
                ch2["text"] = merged
                ch2["char_len"] = len(merged)
                overlapped.append(ch2)
            section_out = overlapped

        chunks.extend(section_out)

    return chunks


def lexical_overlap_score(query: str, text: str) -> float:
    q_terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) > 2]
    if not q_terms:
        return 0.0
    t = text.lower()
    hits = sum(1 for term in q_terms if term in t)
    return hits / len(q_terms)


def load_sentence_transformer(model_name: str, allow_online_model_fetch: bool) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        local_files_only=not allow_online_model_fetch,
    )


def build_index(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf_path)
    out_dir = Path(args.out_dir)
    index_dir = out_dir / "index"
    pages_dir = out_dir / "pages"

    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(pdf_path)
    repeated = detect_repeated_margin_lines(pages)
    cleaned_pages = clean_pages(pages, repeated)
    sections = build_sections(cleaned_pages)

    chunks = build_chunks(
        sections,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
    )

    if not chunks:
        raise RuntimeError("No chunks were produced. Try lowering --min-chars.")

    model = load_sentence_transformer(args.model, args.allow_online_model_fetch)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        chunk_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    np.save(index_dir / "embeddings.npy", embeddings)

    manifest = {
        "pdf_path": str(pdf_path),
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "model": args.model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "min_chars": args.min_chars,
        "pages": len(cleaned_pages),
        "sections": len(sections),
        "chapters": chapter_list_from_sections(sections),
        "repeated_margin_lines_removed": sorted(repeated),
        "segmentation": "hierarchical(part/chapter/section/snippet)",
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (index_dir / "sections.jsonl").open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    for p in cleaned_pages:
        (pages_dir / f"page_{p['page']:04d}.txt").write_text(p["text"] + "\n", encoding="utf-8")

    print(
        f"Indexed {len(cleaned_pages)} pages, {len(sections)} sections, {len(chunks)} chunks at {index_dir} "
        f"(dim={manifest['embedding_dim']})."
    )


def search_index(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    model_name = manifest["model"]

    chunks: list[dict[str, Any]] = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    embeddings = np.load(index_dir / "embeddings.npy")

    model = load_sentence_transformer(model_name, args.allow_online_model_fetch)
    q = model.encode(
        [args.query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    semantic_scores = embeddings @ q

    chapter_filter = args.chapter.strip().lower()
    w_sem = float(args.semantic_weight)
    w_lex = 1.0 - w_sem

    scored: list[tuple[float, int]] = []
    for i, ch in enumerate(chunks):
        if chapter_filter and chapter_filter not in (ch.get("chapter") or "").lower():
            continue

        lex_text = lexical_overlap_score(args.query, ch["text"])
        lex_head = lexical_overlap_score(args.query, ch.get("heading_path") or "")
        meta_boost = 0.08 * lex_head
        score = (w_sem * float(semantic_scores[i])) + (w_lex * lex_text) + meta_boost
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: args.top_k]

    if args.json_output:
        payload: dict[str, Any] = {
            "query": args.query,
            "index": str(index_dir),
            "results": [],
        }
        for score, idx in top:
            ch = chunks[idx]
            payload["results"].append(
                {
                    "score": float(score),
                    "chunk_id": ch["chunk_id"],
                    "page_start": int(ch.get("page_start") or -1),
                    "page_end": int(ch.get("page_end") or -1),
                    "chapter": ch.get("chapter") or "",
                    "section": ch.get("section") or "",
                    "heading_path": ch.get("heading_path") or "",
                    "text": ch["text"],
                }
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(json.dumps({"query": args.query, "index": str(index_dir), "results": len(top)}, indent=2))
    for rank, (score, idx) in enumerate(top, start=1):
        ch = chunks[idx]
        excerpt = re.sub(r"\s+", " ", ch["text"]).strip()
        if len(excerpt) > 280:
            excerpt = excerpt[:280] + "..."
        print(f"[{rank}] score={score:.4f} pages={ch.get('page_start')}-{ch.get('page_end')} chunk_id={ch['chunk_id']}")
        if ch.get("heading_path"):
            print(f"    heading: {ch['heading_path']}")
        print(f"    {excerpt}")


def main() -> None:
    args = parse_args()
    if args.cmd == "build":
        build_index(args)
    elif args.cmd == "search":
        search_index(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
