#!/usr/bin/env python3
"""AFML -> OpenQuant docs authoring loop.

Builds and tracks a hierarchical documentation plan:
- High-level chapters
- Chapter -> module sections
- Section-level writing prompts

This script intentionally calls AFML MCP tool functions from scripts/afml_mcp_server.py
so chapter/module prompts stay grounded in the indexed AFML corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reuse MCP tool functions directly from the local server module.
# This keeps one source of retrieval logic for both server and offline orchestration.
from afml_mcp_server import afml_get_context, afml_list_chapters, afml_search

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "docs" / "afml-docs-loop"
STATE_PATH = STATE_DIR / "state.json"
PROMPTS_DIR = STATE_DIR / "prompts"
MODULE_DOCS_PATH = REPO_ROOT / "docs-site" / "src" / "data" / "moduleDocs.ts"
ASTRO_AFML_DATA_PATH = REPO_ROOT / "docs-site" / "src" / "data" / "afmlDocsState.ts"
EVIDENCE_JSON_PATH = STATE_DIR / "evidence.json"
EVIDENCE_MD_PATH = STATE_DIR / "evidence.md"


@dataclass(frozen=True)
class ChapterBlueprint:
    chapter: str
    theme: str
    query: str
    modules: tuple[str, ...]


BLUEPRINT: tuple[ChapterBlueprint, ...] = (
    ChapterBlueprint("CHAPTER 2", "Event-driven market data", "Bars and sampling features", ("data_structures", "filters", "etf_trick")),
    ChapterBlueprint("CHAPTER 3", "Labeling and target engineering", "Triple-Barrier Method meta-labeling", ("labeling", "bet_sizing")),
    ChapterBlueprint("CHAPTER 4", "Sample uniqueness and weighting", "Average uniqueness sequential bootstrap", ("sampling", "sample_weights", "sb_bagging")),
    ChapterBlueprint("CHAPTER 5", "Stationarity and memory", "Fractional differentiation", ("fracdiff",)),
    ChapterBlueprint("CHAPTER 7", "Leakage-aware validation", "Purged K-Fold CV embargo", ("cross_validation",)),
    ChapterBlueprint("CHAPTER 8", "Feature diagnostics", "Feature Importance MDI MDA", ("feature_importance", "fingerprint")),
    ChapterBlueprint("CHAPTER 10", "Position sizing", "Bet Sizing from Predicted Probabilities", ("bet_sizing",)),
    ChapterBlueprint("CHAPTER 14", "Backtest diagnostics", "Deflated Sharpe Ratio probabilistic Sharpe", ("backtest_statistics", "risk_metrics")),
    ChapterBlueprint("CHAPTER 16", "Portfolio construction", "Hierarchical relationships recursive bisection", ("hrp", "hcaa", "onc", "cla", "portfolio_optimization")),
    ChapterBlueprint("CHAPTER 17", "Regime detection", "Structural break tests SADF CUSUM", ("structural_breaks",)),
    ChapterBlueprint("CHAPTER 18", "Entropy methods", "Shannon entropy Lempel-Ziv", ("microstructural_features",)),
    ChapterBlueprint("CHAPTER 19", "Microstructure estimators", "Kyle lambda Amihud Hasbrouck VPIN", ("microstructural_features", "codependence")),
)

MODULE_QUERIES: dict[str, str] = {
    "data_structures": "Standard Bars time bars imbalance bars run bars",
    "filters": "CUSUM filter event-based sampling",
    "etf_trick": "ETF Trick single future roll",
    "labeling": "Triple-Barrier Method meta-labeling",
    "bet_sizing": "Bet Sizing from Predicted Probabilities averaging active bets",
    "sampling": "Average uniqueness sequential bootstrap",
    "sample_weights": "Return attribution time decay class weights",
    "sb_bagging": "Bagging classifiers uniqueness sequential bootstrap",
    "fracdiff": "fractional differentiation stationarity memory",
    "cross_validation": "Purged K-Fold CV embargo",
    "feature_importance": "Feature Importance MDI MDA",
    "fingerprint": "feature effects substitution effects",
    "backtest_statistics": "Deflated Sharpe Ratio probabilistic Sharpe drawdown",
    "risk_metrics": "value at risk expected shortfall drawdown risk",
    "hrp": "hierarchical relationships recursive bisection",
    "hcaa": "hierarchical clustering based asset allocation",
    "onc": "optimal number of clusters",
    "cla": "critical line algorithm mean-variance constraints",
    "portfolio_optimization": "Markowitz portfolio optimization covariance",
    "structural_breaks": "CUSUM tests explosiveness tests SADF",
    "microstructural_features": "Kyle lambda Amihud Hasbrouck VPIN Shannon entropy",
    "codependence": "mutual information variation of information dependence",
}


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def load_module_map() -> dict[str, str]:
    """Map module name -> slug from docs-site/src/data/moduleDocs.ts."""
    text = MODULE_DOCS_PATH.read_text(encoding="utf-8")
    pairs = re.findall(r"slug:\s*\"([^\"]+)\"\s*,\s*module:\s*\"([^\"]+)\"", text, flags=re.DOTALL)
    mapping = {module: slug for slug, module in pairs}
    if not mapping:
        raise RuntimeError(f"No module docs parsed from {MODULE_DOCS_PATH}")
    return mapping


def summarize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "chapter": hit.get("chapter", ""),
        "section": hit.get("section", ""),
        "page_start": hit.get("page_start", -1),
        "page_end": hit.get("page_end", -1),
        "heading_path": hit.get("heading_path", ""),
        "score": round(float(hit.get("score", 0.0)), 6),
    }


def chapter_eq(left: str, right: str) -> bool:
    def _norm(v: str) -> str:
        return re.sub(r"\s+", " ", v.strip().upper())

    return _norm(left) == _norm(right)


def filter_hits_to_chapter(hits: list[dict[str, Any]], chapter: str) -> list[dict[str, Any]]:
    return [hit for hit in hits if chapter_eq(str(hit.get("chapter", "")), chapter)]


def pick_hit_in_chapter(results: list[dict[str, Any]], chapter: str) -> dict[str, Any]:
    target = chapter.strip().upper()
    for hit in results:
        if str(hit.get("chapter", "")).strip().upper() == target:
            return hit
    return {}


def chapter_prompt(
    chapter: str,
    module: str,
    slug: str,
    module_query: str,
    chapter_hit: dict[str, Any],
    module_hit: dict[str, Any],
) -> str:
    return f"""# Docs Writing Prompt: {chapter} -> {module}

## Scope
- Chapter: `{chapter}`
- Module: `{module}` (`/module/{slug}`)
- Source files:
  - `docs-site/src/data/moduleDocs.ts` (module content)
  - `docs-site/src/pages/modules.astro` (subject/chapter entry point)
  - `docs-site/src/pages/module/[slug].astro` (module page rendering)

## Required AFML grounding
Use AFML MCP retrieval before writing:
1. `afml_search(query=\"{chapter_hit.get('section','')}\", chapter=\"{chapter}\", top_k=3)`
2. `afml_search(query=\"{module_query}\", chapter=\"{chapter}\", top_k=3)`
3. `afml_get_context(query=\"{module_query}\", chapter=\"{chapter}\", top_k=2, window=1)`

Use chunk ids and page ranges in your internal notes.

## Content contract (must be complete)
1. High-level purpose and failure modes.
2. AFML math framing tied to API behavior.
3. Minimum 2 executable Rust examples (realistic, not toy-only).
4. Implementation caveats and production constraints.
5. Cross-links to adjacent modules in same chapter/subject.

## UI/UX contract (not AI slop)
1. Keep typography intentional and consistent.
2. Improve scanability with clear section rhythm and spacing.
3. Avoid generic gradient-heavy cards everywhere.
4. Keep code blocks readable on mobile and desktop.
5. Verify rendered page in Astro before marking complete.

## Current AFML anchors
- Chapter anchor: `{chapter_hit.get('chunk_id','')}` pages {chapter_hit.get('page_start',-1)}-{chapter_hit.get('page_end',-1)} ({chapter_hit.get('section','')})
- Module anchor: `{module_hit.get('chunk_id','')}` pages {module_hit.get('page_start',-1)}-{module_hit.get('page_end',-1)} ({module_hit.get('section','')})
"""


def build_state() -> dict[str, Any]:
    chapters = afml_list_chapters()["chapters"]
    chapter_counts = {item["chapter"]: item["chunk_count"] for item in chapters}
    module_map = load_module_map()

    chapter_nodes: list[dict[str, Any]] = []
    for blueprint in BLUEPRINT:
        chapter_search = afml_search(query=blueprint.query, chapter=blueprint.chapter, top_k=5, include_text=False)
        chapter_hit_raw = pick_hit_in_chapter(chapter_search["results"], blueprint.chapter)
        chapter_hit = summarize_hit(chapter_hit_raw) if chapter_hit_raw else {}

        module_nodes: list[dict[str, Any]] = []
        for module in blueprint.modules:
            slug = module_map.get(module)
            if slug is None:
                slug = module.replace("::", "-").replace("_", "-")

            module_query = MODULE_QUERIES.get(module, module.replace("_", " "))
            module_search = afml_search(query=module_query, chapter=blueprint.chapter, top_k=5, include_text=False)
            module_hit_raw = pick_hit_in_chapter(module_search["results"], blueprint.chapter)
            module_hit = summarize_hit(module_hit_raw) if module_hit_raw else {}

            section_id = f"{blueprint.chapter.lower().replace(' ', '-')}-{module}"
            module_nodes.append(
                {
                    "id": section_id,
                    "module": module,
                    "slug": slug,
                    "status": "pending",
                    "afml_anchor": module_hit,
                }
            )

            prompt_body = chapter_prompt(blueprint.chapter, module, slug, module_query, chapter_hit, module_hit)
            PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
            (PROMPTS_DIR / f"{section_id}.md").write_text(prompt_body, encoding="utf-8")

        chapter_nodes.append(
            {
                "chapter": blueprint.chapter,
                "theme": blueprint.theme,
                "query": blueprint.query,
                "chunk_count": chapter_counts.get(blueprint.chapter, 0),
                "status": "pending",
                "afml_anchor": chapter_hit,
                "sections": module_nodes,
            }
        )

    return {
        "generated_at": now_iso(),
        "updated_at": now_iso(),
        "state_version": 1,
        "overview": {
            "astro_module_data": str(MODULE_DOCS_PATH.relative_to(REPO_ROOT)),
            "module_page": "docs-site/src/pages/module/[slug].astro",
            "subjects_page": "docs-site/src/pages/modules.astro",
        },
        "chapters": chapter_nodes,
    }


def write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        raise FileNotFoundError("State not found. Run: scripts/afml_docs_loop.py init")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def recompute_chapter_status(chapter: dict[str, Any]) -> str:
    statuses = {s["status"] for s in chapter["sections"]}
    if statuses == {"done"}:
        return "done"
    if "in_progress" in statuses or ("done" in statuses and "pending" in statuses):
        return "in_progress"
    return "pending"


def iter_sections(state: dict[str, Any]):
    for chapter in state["chapters"]:
        for section in chapter["sections"]:
            yield chapter, section


def cmd_init(_args: argparse.Namespace) -> int:
    state = build_state()
    write_state(state)
    print(f"Initialized docs loop state: {STATE_PATH}")
    print(f"Prompt files: {PROMPTS_DIR}")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    state = read_state()
    total = 0
    done = 0
    in_progress = 0
    for chapter in state["chapters"]:
        chapter["status"] = recompute_chapter_status(chapter)
        for section in chapter["sections"]:
            total += 1
            if section["status"] == "done":
                done += 1
            elif section["status"] == "in_progress":
                in_progress += 1

    pending = total - done - in_progress
    print(f"Total sections: {total} | done: {done} | in_progress: {in_progress} | pending: {pending}")
    for chapter in state["chapters"]:
        chapter_done = sum(1 for s in chapter["sections"] if s["status"] == "done")
        print(f"- {chapter['chapter']} [{chapter['status']}] {chapter_done}/{len(chapter['sections'])} :: {chapter['theme']}")
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    state = read_state()
    for chapter, section in iter_sections(state):
        if section["status"] == "pending":
            section["status"] = "in_progress"
            chapter["status"] = recompute_chapter_status(chapter)
            write_state(state)

            prompt_path = PROMPTS_DIR / f"{section['id']}.md"
            print(f"Next section: {section['id']}")
            print(f"Chapter: {chapter['chapter']} | Module: {section['module']} | Slug: {section['slug']}")
            print(f"Prompt file: {prompt_path}")
            if args.print_prompt:
                print("\n" + prompt_path.read_text(encoding="utf-8"))
            return 0
    print("No pending sections. All sections are done.")
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    state = read_state()
    target_id = args.section_id
    found = False
    for chapter, section in iter_sections(state):
        if section["id"] == target_id:
            found = True
            section["status"] = "done"
            if args.note:
                section["note"] = args.note
            chapter["status"] = recompute_chapter_status(chapter)
            break

    if not found:
        raise ValueError(f"Unknown section id: {target_id}")

    write_state(state)
    print(f"Marked done: {target_id}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    state = read_state()
    target_id = args.section_id
    found = False
    for chapter, section in iter_sections(state):
        if section["id"] == target_id:
            found = True
            section["status"] = "pending"
            section.pop("note", None)
            chapter["status"] = recompute_chapter_status(chapter)
            break
    if not found:
        raise ValueError(f"Unknown section id: {target_id}")

    write_state(state)
    print(f"Reset to pending: {target_id}")
    return 0


def cmd_export(_args: argparse.Namespace) -> int:
    state = read_state()

    export = {
        "generatedAt": state.get("updated_at", now_iso()),
        "chapters": [],
    }
    for chapter in state["chapters"]:
        export["chapters"].append(
            {
                "chapter": chapter["chapter"],
                "theme": chapter["theme"],
                "status": recompute_chapter_status(chapter),
                "chunkCount": chapter.get("chunk_count", 0),
                "sections": [
                    {
                        "id": section["id"],
                        "module": section["module"],
                        "slug": section["slug"],
                        "status": section["status"],
                    }
                    for section in chapter["sections"]
                ],
            }
        )

    body = (
        "// Generated by scripts/afml_docs_loop.py export. Do not edit manually.\n"
        f"export const afmlDocsState = {json.dumps(export, indent=2)} as const;\n"
        "export type AfmlDocsState = typeof afmlDocsState;\n"
    )
    ASTRO_AFML_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASTRO_AFML_DATA_PATH.write_text(body, encoding="utf-8")
    print(f"Exported Astro AFML docs state: {ASTRO_AFML_DATA_PATH}")
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    state = read_state()
    evidence_chapters: list[dict[str, Any]] = []
    processed = 0

    for chapter in state["chapters"]:
        chapter_name = chapter["chapter"]
        chapter_query = chapter.get("query") or chapter.get("theme") or chapter_name
        chapter_search = afml_search(
            query=chapter_query,
            chapter=chapter_name,
            top_k=args.top_k,
            include_text=False,
        )
        chapter_hits = filter_hits_to_chapter(chapter_search.get("results", []), chapter_name)
        chapter_anchor = summarize_hit(chapter_hits[0]) if chapter_hits else chapter.get("afml_anchor", {})
        chapter["afml_anchor"] = chapter_anchor

        chapter_evidence: dict[str, Any] = {
            "chapter": chapter_name,
            "query": chapter_query,
            "anchor": chapter_anchor,
            "sections": [],
        }

        for section in chapter["sections"]:
            if args.only_pending and section["status"] != "pending":
                continue
            processed += 1

            module = section["module"]
            module_query = MODULE_QUERIES.get(module, module.replace("_", " "))
            search_out = afml_search(
                query=module_query,
                chapter=chapter_name,
                top_k=args.top_k,
                include_text=False,
            )
            search_hits = filter_hits_to_chapter(search_out.get("results", []), chapter_name)

            ctx_out = afml_get_context(
                query=module_query,
                chapter=chapter_name,
                top_k=args.top_k_context,
                window=args.window,
            )
            contexts = [
                c for c in ctx_out.get("contexts", []) if chapter_eq(str(c.get("hit", {}).get("chapter", "")), chapter_name)
            ]

            primary_hit: dict[str, Any] = {}
            if contexts:
                primary_hit = contexts[0].get("hit", {})
            elif search_hits:
                primary_hit = search_hits[0]

            if primary_hit:
                section["afml_anchor"] = summarize_hit(primary_hit)

            section["evidence"] = {
                "retrieved_at": now_iso(),
                "module_query": module_query,
                "search_hits": [summarize_hit(h) for h in search_hits[: args.keep_hits]],
                "context_hits": [summarize_hit(c.get("hit", {})) for c in contexts[: args.keep_hits]],
            }
            chapter_evidence["sections"].append(
                {
                    "id": section["id"],
                    "module": module,
                    "slug": section["slug"],
                    "status": section["status"],
                    "module_query": module_query,
                    "anchor": section.get("afml_anchor", {}),
                    "search_hits": section["evidence"]["search_hits"],
                    "context_hits": section["evidence"]["context_hits"],
                }
            )

            if args.mark_done:
                section["status"] = "done"
                section["note"] = "Completed via MCP evidence loop"

        chapter["status"] = recompute_chapter_status(chapter)
        evidence_chapters.append(chapter_evidence)

    write_state(state)

    evidence_payload = {
        "generated_at": now_iso(),
        "top_k": args.top_k,
        "top_k_context": args.top_k_context,
        "window": args.window,
        "chapters": evidence_chapters,
    }
    EVIDENCE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON_PATH.write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")

    md_lines = [
        "# AFML MCP Evidence",
        "",
        f"- generated_at: `{evidence_payload['generated_at']}`",
        f"- top_k: `{args.top_k}`",
        f"- top_k_context: `{args.top_k_context}`",
        f"- window: `{args.window}`",
        "",
    ]
    for ch in evidence_chapters:
        md_lines.append(f"## {ch['chapter']}")
        anchor = ch.get("anchor", {})
        if anchor:
            md_lines.append(
                f"- chapter_anchor: `{anchor.get('chunk_id','')}` ({anchor.get('section','')}) pages {anchor.get('page_start',-1)}-{anchor.get('page_end',-1)}"
            )
        else:
            md_lines.append("- chapter_anchor: `(none)`")
        md_lines.append("")
        for sec in ch["sections"]:
            a = sec.get("anchor", {})
            md_lines.append(f"### {sec['id']}")
            md_lines.append(f"- module: `{sec['module']}` (`/module/{sec['slug']}`)")
            md_lines.append(f"- query: `{sec['module_query']}`")
            if a:
                md_lines.append(
                    f"- anchor: `{a.get('chunk_id','')}` ({a.get('section','')}) pages {a.get('page_start',-1)}-{a.get('page_end',-1)}"
                )
            else:
                md_lines.append("- anchor: `(none)`")
            md_lines.append("")

    EVIDENCE_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote MCP evidence: {EVIDENCE_JSON_PATH}")
    print(f"Wrote MCP evidence summary: {EVIDENCE_MD_PATH}")
    print(f"Sections processed: {processed}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFML docs hierarchy authoring loop")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Generate state + writing prompts")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="Show progress")
    p_status.set_defaults(func=cmd_status)

    p_next = sub.add_parser("next", help="Claim next pending section")
    p_next.add_argument("--print-prompt", action="store_true", help="Print prompt body")
    p_next.set_defaults(func=cmd_next)

    p_done = sub.add_parser("done", help="Mark section done")
    p_done.add_argument("section_id")
    p_done.add_argument("--note", default="", help="Optional completion note")
    p_done.set_defaults(func=cmd_done)

    p_reset = sub.add_parser("reset", help="Reset section to pending")
    p_reset.add_argument("section_id")
    p_reset.set_defaults(func=cmd_reset)

    p_export = sub.add_parser("export", help="Export state into docs-site data for UI rendering")
    p_export.set_defaults(func=cmd_export)

    p_evidence = sub.add_parser("evidence", help="Run MCP retrieval loop and persist per-section evidence")
    p_evidence.add_argument("--top-k", type=int, default=5, help="Top K hits for afml_search")
    p_evidence.add_argument("--top-k-context", type=int, default=3, help="Top K hits for afml_get_context")
    p_evidence.add_argument("--window", type=int, default=1, help="Neighbor chunk window for afml_get_context")
    p_evidence.add_argument("--keep-hits", type=int, default=3, help="Number of hits to persist per section")
    p_evidence.add_argument("--only-pending", action="store_true", help="Process only sections still pending")
    p_evidence.add_argument("--mark-done", action="store_true", help="Mark processed sections done")
    p_evidence.set_defaults(func=cmd_evidence)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
