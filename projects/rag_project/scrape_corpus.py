#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape Spanish news / reports pages into a JSONL corpus, ready for RAG.
- Pulls from RSS/Atom feeds (via feedparser) and/or a list of direct URLs.
- Extracts main article text + metadata with trafilatura (robust for ES content).
- Writes one JSON object per line to ./projects/rag_project/corpus/articles.jsonl

Why this file?
- Twitter scraping via snscrape is currently blocked by X endpoint changes (404).
- This is a drop-in alternative to build your Spanish corpus from open web sources
  (news, NGO reports, blogs, think-tanks, etc.) for your RAG pipeline.

Dependencies (install once in your venv):
  pip install trafilatura feedparser

Usage examples:
  python projects/rag_project/scrape_corpus.py
  python projects/rag_project/scrape_corpus.py --feeds https://example.com/rss https://another.com/feed
  python projects/rag_project/scrape_corpus.py --urls https://som360.org/... https://elpais.com/...
  python projects/rag_project/scrape_corpus.py --feeds-file projects/rag_project/feeds.txt --urls-file projects/rag_project/urls.txt --max-per-feed 50

Notes:
- Respect robots.txt and website terms. Use moderate rates (--sleep).
- If some media are paywalled, you may only be able to capture previews.
- Feeds are more stable than ad-hoc crawling; prefer RSS/Atom when possible.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Iterable, List, Optional, Set
from urllib.parse import urlparse

import feedparser  # pip install feedparser
import trafilatura  # pip install trafilatura
import requests  # HTTP fallback with custom UA


# Default seeds (edit as needed). You can override with CLI flags/files.
DEFAULT_FEEDS: List[str] = [
    # Add RSS/Atom feeds here (examples below may need updating by site)
    # Suggestion examples (verify actual RSS URLs in browser):
    # "https://som360.org/es/rss",
    # "https://elpais.com/rss/elpais/planeta_futuro.xml",
    # "https://www.infobae.com/feeds/rss/",
]
DEFAULT_URLS: List[str] = [
    # Add direct article/report URLs here if you don't have feeds.
    # "https://som360.org/es/temas/adolescencia",
    # "https://elpais.com/planeta-futuro/",
]


def read_lines_file(path: str) -> List[str]:
    items: List[str] = []
    if not path or not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def ensure_outdir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def extract_with_trafilatura(url: str) -> Optional[dict]:
    """
    Returns a dict with keys: url, title, author, date, text, source, language
    or None if extraction fails.
    Tries trafilatura.fetch_url with a browser-like UA; falls back to requests.get.
    """
    UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    html: Optional[str] = None

    # Try trafilatura with custom config (browser UA)
    try:
        from trafilatura.settings import use_config
        cfg = use_config()
        cfg.set("DEFAULT", "USER_AGENT", UA)
        # Be lenient on minimum size/timeouts
        cfg.set("DEFAULT", "MIN_EXTRACTED_SIZE", "0")
        cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        raw = trafilatura.fetch_url(url, config=cfg)
        if raw:
            html = raw
    except Exception as e:
        print(f"[DEBUG] trafilatura.fetch_url error for {url}: {e}")

    # Fallback: requests with UA
    if html is None:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": UA, "Accept-Language": "es-ES,es;q=0.9"},
                timeout=20,
            )
            if resp.status_code == 200 and resp.text:
                html = resp.text
            else:
                print(f"[DEBUG] requests.get status={resp.status_code} url={url}")
        except Exception as e:
            print(f"[DEBUG] requests.get error for {url}: {e}")
            html = None

    if not html:
        return None

    # Attempt rich JSON extraction (with metadata)
    try:
        extracted_json = trafilatura.extract(
            html,
            output_format="json",
            include_comments=False,
            include_tables=False,
            favor_recall=False,   # prefer precision for articles
            favor_precision=True,
            with_metadata=True,
            deduplicate=True,
            target_language="es",  # hint for ES
        )
        if extracted_json:
            data = json.loads(extracted_json)
            return {
                "url": url,
                "title": data.get("title"),
                "author": data.get("author"),
                "date": data.get("date"),  # ISO if available
                "text": data.get("text"),
                "source": urlparse(url).netloc,
                "language": data.get("language"),
            }
    except Exception as e:
        print(f"[DEBUG] trafilatura.extract(json) error for {url}: {e}")

    # Fallback: plain text only
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=False,
            favor_precision=True,
            deduplicate=True,
            target_language="es",
        )
    except Exception as e:
        print(f"[DEBUG] trafilatura.extract(text) error for {url}: {e}")
        text = None

    if not text:
        return None

    return {
        "url": url,
        "title": None,
        "author": None,
        "date": None,
        "text": text,
        "source": urlparse(url).netloc,
        "language": "es",
    }


def iter_feed_links(feed_url: str, max_items: int) -> Iterable[str]:
    parsed = feedparser.parse(feed_url)
    if parsed.bozo:
        return []
    count = 0
    for entry in parsed.entries:
        link = getattr(entry, "link", None)
        if not link:
            continue
        yield link
        count += 1
        if count >= max_items:
            break


def write_jsonl(records: Iterable[dict], out_path: str) -> int:
    ensure_outdir(out_path)
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for rec in records:
            if not rec or not rec.get("text"):
                continue
            # Add timestamp and id
            rec_out = dict(rec)
            rec_out["scraped_at"] = datetime.utcnow().isoformat() + "Z"
            rec_out["id"] = f'{rec_out.get("url","")}|{rec_out["scraped_at"]}'
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description="Scrape Spanish articles into JSONL corpus.")
    ap.add_argument("--feeds", nargs="*", default=None, help="List of feed URLs (RSS/Atom).")
    ap.add_argument("--urls", nargs="*", default=None, help="List of direct article URLs.")
    ap.add_argument("--feeds-file", type=str, default=None, help="Path to a file with feed URLs (one per line).")
    ap.add_argument("--urls-file", type=str, default=None, help="Path to a file with article URLs (one per line).")
    ap.add_argument("--out", type=str, default="projects/rag_project/corpus/articles.jsonl", help="Output JSONL path.")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between fetches.")
    ap.add_argument("--max-per-feed", type=int, default=50, help="Max items to pull from each feed.")
    args = ap.parse_args()

    # Compose effective lists
    feeds: List[str] = []
    urls: List[str] = []

    if args.feeds is not None:
        feeds.extend(args.feeds)
    if args.urls is not None:
        urls.extend(args.urls)
    if args.feeds_file:
        feeds.extend(read_lines_file(args.feeds_file))
    if args.urls_file:
        urls.extend(read_lines_file(args.urls_file))

    if not feeds and not urls:
        # Use defaults if nothing is provided
        feeds = list(DEFAULT_FEEDS)
        urls = list(DEFAULT_URLS)

    # Deduplicate while preserving input order
    def dedup(seq: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    feeds = dedup([s for s in feeds if s])
    urls = dedup([s for s in urls if s])

    print(f"[INFO] Feeds: {len(feeds)}  URLs: {len(urls)}  -> Output: {args.out}")

    total_written = 0
    seen_urls: Set[str] = set()

    # 1) From feeds
    for feed in feeds:
        print(f"[INFO] Parsing feed: {feed}")
        try:
            for link in iter_feed_links(feed, max_items=args.max_per_feed):
                if link in seen_urls:
                    continue
                seen_urls.add(link)
                rec = extract_with_trafilatura(link)
                if rec:
                    total_written += write_jsonl([rec], args.out)
                    print(f"[OK] {link}")
                else:
                    print(f"[WARN] Failed extract: {link}")
                time.sleep(args.sleep)
        except Exception as e:
            print(f"[ERROR] Feed error {feed}: {e}")

    # 2) From direct URLs
    for link in urls:
        if link in seen_urls:
            continue
        seen_urls.add(link)
        rec = extract_with_trafilatura(link)
        if rec:
            total_written += write_jsonl([rec], args.out)
            print(f"[OK] {link}")
        else:
            print(f"[WARN] Failed extract: {link}")
        time.sleep(args.sleep)

    print(f"[DONE] Wrote {total_written} records to {args.out}")


if __name__ == "__main__":
    main()
