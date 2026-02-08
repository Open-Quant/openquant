#!/usr/bin/env python3
"""MCP server for AFML semantic retrieval.

Default transport: stdio

Environment variables:
- AFML_INDEX_DIR: path to index directory (default: afml/doc_1/index)
- AFML_ALLOW_ONLINE_MODEL_FETCH: '1' to allow HF online fetch, else local-only
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from afml_semantic_index import lexical_overlap_score, load_sentence_transformer


@dataclass
class IndexStore:
    index_dir: Path
    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    manifest: dict[str, Any]
    chunk_id_to_idx: dict[str, int]


_STORE: IndexStore | None = None
_MODEL = None


def _allow_online_fetch() -> bool:
    return os.getenv("AFML_ALLOW_ONLINE_MODEL_FETCH", "0") in {"1", "true", "TRUE"}


def _get_index_dir() -> Path:
    return Path(os.getenv("AFML_INDEX_DIR", "afml/doc_1/index"))


def _load_store() -> IndexStore:
    global _STORE
    if _STORE is not None:
        return _STORE

    index_dir = _get_index_dir()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

    chunks: list[dict[str, Any]] = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    embeddings = np.load(index_dir / "embeddings.npy")
    chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    _STORE = IndexStore(
        index_dir=index_dir,
        chunks=chunks,
        embeddings=embeddings,
        manifest=manifest,
        chunk_id_to_idx=chunk_id_to_idx,
    )
    return _STORE


def _load_model(model_name: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = load_sentence_transformer(model_name, _allow_online_fetch())
    return _MODEL


def _score_results(
    query: str,
    top_k: int,
    chapter: str = "",
    section: str = "",
    semantic_weight: float = 0.86,
) -> list[dict[str, Any]]:
    store = _load_store()
    model = _load_model(store.manifest["model"])

    q = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    semantic_scores = store.embeddings @ q

    chapter_filter = chapter.strip().lower()
    section_filter = section.strip().lower()
    w_sem = float(semantic_weight)
    w_lex = 1.0 - w_sem

    scored: list[tuple[float, int]] = []
    for i, ch in enumerate(store.chunks):
        ch_chapter = (ch.get("chapter") or "").lower()
        ch_section = (ch.get("section") or "").lower()
        if chapter_filter and chapter_filter not in ch_chapter:
            continue
        if section_filter and section_filter not in ch_section:
            continue

        lex_text = lexical_overlap_score(query, ch["text"])
        lex_head = lexical_overlap_score(query, ch.get("heading_path") or "")
        meta_boost = 0.08 * lex_head
        score = (w_sem * float(semantic_scores[i])) + (w_lex * lex_text) + meta_boost
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for score, idx in scored[:top_k]:
        ch = store.chunks[idx]
        out.append(
            {
                "score": float(score),
                "chunk_id": ch["chunk_id"],
                "section_id": ch.get("section_id"),
                "page_start": int(ch.get("page_start") or -1),
                "page_end": int(ch.get("page_end") or -1),
                "chapter": ch.get("chapter") or "",
                "section": ch.get("section") or "",
                "heading_path": ch.get("heading_path") or "",
                "text": ch["text"],
            }
        )
    return out


def _chunk_with_neighbors(chunk_id: str, window: int) -> dict[str, Any]:
    store = _load_store()
    idx = store.chunk_id_to_idx.get(chunk_id)
    if idx is None:
        raise ValueError(f"Unknown chunk_id: {chunk_id}")

    start = max(0, idx - window)
    end = min(len(store.chunks), idx + window + 1)

    center = store.chunks[idx]
    neighbors = store.chunks[start:end]
    return {
        "center": center,
        "neighbors": neighbors,
    }


mcp = FastMCP(
    name="AFML Quant Research Server",
    instructions=(
        "Semantic retrieval over AFML literature for quant research. "
        "Prefer afml_search for discovery, then afml_get_chunk or afml_get_context for deeper reading."
    ),
)


@mcp.tool(name="afml_index_info", description="Return index metadata, chapters, and model details.")
def afml_index_info() -> dict[str, Any]:
    store = _load_store()
    return {
        "index_dir": str(store.index_dir),
        "manifest": store.manifest,
        "chunk_count": len(store.chunks),
    }


@mcp.tool(name="afml_list_chapters", description="List discovered chapters with chunk counts.")
def afml_list_chapters() -> dict[str, Any]:
    store = _load_store()
    counts: dict[str, int] = {}
    for ch in store.chunks:
        key = (ch.get("chapter") or "(no chapter)").strip()
        counts[key] = counts.get(key, 0) + 1

    items = [{"chapter": k, "chunk_count": v} for k, v in sorted(counts.items(), key=lambda x: x[0])]
    return {"chapters": items}


@mcp.tool(
    name="afml_search",
    description=(
        "Semantic + lexical retrieval for AFML chunks. "
        "Use chapter/section filters for focused quant research."
    ),
)
def afml_search(
    query: str,
    top_k: int = 8,
    chapter: str = "",
    section: str = "",
    semantic_weight: float = 0.86,
    include_text: bool = True,
) -> dict[str, Any]:
    if top_k < 1:
        top_k = 1
    if top_k > 50:
        top_k = 50

    results = _score_results(
        query=query,
        top_k=top_k,
        chapter=chapter,
        section=section,
        semantic_weight=semantic_weight,
    )
    if not include_text:
        for r in results:
            r.pop("text", None)

    return {
        "query": query,
        "top_k": top_k,
        "filters": {"chapter": chapter, "section": section},
        "results": results,
    }


@mcp.tool(
    name="afml_get_chunk",
    description="Fetch a chunk by chunk_id, optionally including neighboring chunks for context.",
)
def afml_get_chunk(chunk_id: str, window: int = 0) -> dict[str, Any]:
    if window < 0:
        window = 0
    if window > 5:
        window = 5
    return _chunk_with_neighbors(chunk_id=chunk_id, window=window)


@mcp.tool(
    name="afml_get_context",
    description=(
        "Search then return each hit with neighbor windows. "
        "Useful for agents drafting grounded quant-research notes from AFML."
    ),
)
def afml_get_context(
    query: str,
    top_k: int = 4,
    window: int = 1,
    chapter: str = "",
    section: str = "",
    semantic_weight: float = 0.86,
) -> dict[str, Any]:
    if window < 0:
        window = 0
    if window > 5:
        window = 5

    hits = _score_results(
        query=query,
        top_k=top_k,
        chapter=chapter,
        section=section,
        semantic_weight=semantic_weight,
    )

    contexts: list[dict[str, Any]] = []
    for hit in hits:
        bundle = _chunk_with_neighbors(hit["chunk_id"], window=window)
        contexts.append({"hit": hit, "context": bundle})

    return {
        "query": query,
        "top_k": top_k,
        "window": window,
        "filters": {"chapter": chapter, "section": section},
        "contexts": contexts,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
