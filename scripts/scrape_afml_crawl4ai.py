#!/usr/bin/env python3
"""Scrape AFML FlipHTML5 pages with Crawl4AI.

Example:
  python scripts/scrape_afml_crawl4ai.py \
    --start-url 'https://fliphtml5.com/fzqli/zwcp/Advances_in_Financial_Machine_Learning-Marcos_Lopez_de_Prado/236/' \
    --start-page 236 \
    --end-page 240 \
    --output-root afml
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape AFML pages from FlipHTML5 into afml/doc_<n>/ folders"
    )
    parser.add_argument(
        "--start-url",
        default=(
            "https://fliphtml5.com/fzqli/zwcp/"
            "Advances_in_Financial_Machine_Learning-Marcos_Lopez_de_Prado/236/"
        ),
        help="A FlipHTML5 page URL that includes a trailing page number",
    )
    parser.add_argument("--start-page", type=int, default=236)
    parser.add_argument("--end-page", type=int, default=236)
    parser.add_argument("--output-root", default="afml")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=4.0,
        help="Extra JS settle time per page",
    )
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run browser in headed mode",
    )
    return parser.parse_args()


def derive_base_url(start_url: str) -> str:
    m = re.match(r"^(.*?/)(\d+)/?$", start_url.strip())
    if not m:
        raise ValueError(
            "--start-url must end with a numeric page, e.g. .../236/"
        )
    return m.group(1)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    # Crawl4AI markdown fields can be objects depending on version.
    for attr in ("raw_markdown", "fit_markdown", "markdown", "text", "content"):
        if hasattr(value, attr):
            attr_val = getattr(value, attr)
            if isinstance(attr_val, str) and attr_val.strip():
                return attr_val

    try:
        return str(value)
    except Exception:
        return ""


def choose_markdown(result: Any) -> str:
    for attr in ("markdown_v2", "markdown"):
        if hasattr(result, attr):
            text = to_text(getattr(result, attr))
            if text.strip():
                return text
    return ""


def choose_html(result: Any) -> str:
    for attr in ("cleaned_html", "html"):
        if hasattr(result, attr):
            text = to_text(getattr(result, attr))
            if text.strip():
                return text
    return ""


def safe_metadata(result: Any, url: str, page: int, doc_id: int) -> dict[str, Any]:
    metadata = {
        "url": url,
        "page": page,
        "doc_id": doc_id,
    }
    for attr in ("success", "status_code", "error_message", "response_headers"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            try:
                json.dumps(value)
                metadata[attr] = value
            except Exception:
                metadata[attr] = str(value)
    return metadata


async def scrape_range(args: argparse.Namespace) -> None:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    except ImportError as exc:
        raise SystemExit(
            "crawl4ai is not installed. Install it first, e.g. `pip install crawl4ai`."
        ) from exc

    if args.end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page")

    base_url = derive_base_url(args.start_url)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(headless=args.headless)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=args.timeout_ms,
        wait_for=f"js:() => new Promise(r => setTimeout(r, {int(args.wait_seconds * 1000)}))",
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for doc_idx, page in enumerate(range(args.start_page, args.end_page + 1), start=1):
            url = f"{base_url}{page}/"
            doc_dir = output_root / f"doc_{doc_idx}"
            doc_dir.mkdir(parents=True, exist_ok=True)

            result = await crawler.arun(url=url, config=run_config)

            markdown = choose_markdown(result)
            html = choose_html(result)
            metadata = safe_metadata(result, url=url, page=page, doc_id=doc_idx)

            (doc_dir / "source_url.txt").write_text(url + "\n", encoding="utf-8")
            (doc_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            (doc_dir / "content.md").write_text(markdown, encoding="utf-8")
            (doc_dir / "content.html").write_text(html, encoding="utf-8")

            print(
                f"saved {doc_dir} "
                f"(page={page}, markdown_chars={len(markdown)}, html_chars={len(html)})"
            )


async def main_async() -> None:
    args = parse_args()
    await scrape_range(args)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
