#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter/X scraper using snscrape CLI with optional authenticated cookies.

Why this file exists:
- The Python API path (snscrape.modules.twitter.TwitterSearchScraper) often fails with
  "blocked (404)" due to X internal endpoint changes and stricter anti-scraping.
- The CLI + cookies flow is more resilient. You can run without cookies, but if X blocks
  guest traffic, you must supply your browser cookies.

How to supply cookies (recommended):
1) Log in to https://x.com (twitter.com) in your browser.
2) Export cookies using a browser extension (e.g., "Cookie-Editor") as Netscape format.
3) Save to a file, e.g., projects/rag_project/cookies.txt (or any path).
4) Either:
   - Set env var before running: SNSCRAPE_TWITTER_COOKIES=projects/rag_project/cookies.txt
   - Or place cookies.txt next to this script; it will be auto-detected.

Usage examples:
  python projects/rag_project/twitterscrap.py \
    --query "#GeneraciónZ" \
    --since 2024-01-01 --until 2025-12-31 \
    --lang es --limit 1000 \
    --out projects/rag_project/corpus/twitter_es_genz.jsonl

Notes:
- Requires snscrape to be installed in the current environment:
    pip install snscrape
- This script reads snscrape's JSONL stream and truncates to --limit.
- Output is JSONL with selected fields for downstream RAG ingestion.
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def build_query(base: str, lang: str, since: str, until: str) -> str:
    parts = [base.strip()]
    if lang:
        parts.append(f"lang:{lang}")
    if since:
        parts.append(f"since:{since}")
    if until:
        parts.append(f"until:{until}")
    return " ".join(parts)


def ensure_output_dir(path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)


def detect_cookies_env(script_dir: Path) -> dict:
    env = os.environ.copy()
    # If user already exported env var, keep it
    if env.get("SNSCRAPE_TWITTER_COOKIES"):
        return env
    # Auto-detect cookies.txt in repo
    candidates = [
        script_dir / "cookies.txt",
        script_dir.parent / "cookies.txt",
        Path.cwd() / "cookies.txt",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            env["SNSCRAPE_TWITTER_COOKIES"] = str(p)
            print(f"[INFO] Using cookies file: {p}")
            break
    return env


def run_snscrape_stream(query: str, limit: int, env: dict):
    """
    Invoke snscrape CLI and yield tweet JSON objects (dict) up to 'limit'.
    """
    cmd = ["snscrape", "--jsonl", "twitter-search", query]
    print(f"[INFO] Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )
    count = 0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj
            count += 1
            if count >= limit:
                break
    finally:
        # If still running, terminate to avoid scanning whole timeline
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Consume remaining stderr to show helpful diagnostics if nothing was scraped
        if count == 0 and proc.stderr is not None:
            err = proc.stderr.read().strip()
            if err:
                print("[DEBUG snscrape stderr]")
                print(err)


def normalize_tweet(obj: dict) -> dict:
    """
    Reduce snscrape's tweet object to minimal fields for RAG.
    """
    return {
        "id": obj.get("id"),
        "url": obj.get("url"),
        "date": obj.get("date"),
        "content": obj.get("content"),
        "username": (obj.get("user") or {}).get("username") if isinstance(obj.get("user"), dict) else None,
        "replyCount": obj.get("replyCount"),
        "retweetCount": obj.get("retweetCount"),
        "likeCount": obj.get("likeCount"),
        "quoteCount": obj.get("quoteCount"),
        "lang": obj.get("lang"),
        "sourceLabel": obj.get("sourceLabel"),
    }


def main():
    ap = argparse.ArgumentParser(description="Scrape tweets/posts from X via snscrape CLI.")
    ap.add_argument("--query", required=False, default="#GeneraciónZ", help="Base query (e.g., hashtag or keywords).")
    ap.add_argument("--since", required=False, default="2024-01-01", help="Since date YYYY-MM-DD.")
    ap.add_argument("--until", required=False, default="2025-12-31", help="Until date YYYY-MM-DD (exclusive).")
    ap.add_argument("--lang", required=False, default="es", help="Language filter (e.g., es).")
    ap.add_argument("--limit", type=int, default=1000, help="Max number of items to collect.")
    ap.add_argument("--out", required=False, default="projects/rag_project/corpus/twitter_es_genz.jsonl", help="Output JSONL path.")
    args = ap.parse_args()

    query = build_query(args.query, args.lang, args.since, args.until)
    ensure_output_dir(args.out)

    script_dir = Path(__file__).resolve().parent
    env = detect_cookies_env(script_dir)

    collected = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for obj in run_snscrape_stream(query, args.limit, env):
            rec = normalize_tweet(obj)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            collected += 1

    if collected == 0:
        print("[ERROR] No tweets collected. This often means guest requests are blocked by X.")
        print("Fix by exporting cookies and setting SNSCRAPE_TWITTER_COOKIES, e.g.:")
        print("  export SNSCRAPE_TWITTER_COOKIES=projects/rag_project/cookies.txt")
        print("Then rerun this command.")
        sys.exit(1)

    print(f"[DONE] Collected {collected} tweets to {args.out}")
    print(f"[TIP] Next steps:")
    print(f"  - Chunk and embed with your pipeline, e.g.:")
    print(f"    python projects/rag_project/chunk_jsonl.py --input {args.out} --output {Path(args.out).with_suffix('.chunks.jsonl')}")
    print(f"    python projects/rag_project/embed_index_ollama.py --input {Path(args.out).with_suffix('.chunks.jsonl')} --persist-dir projects/rag_project/vectorstore --collection genz_corpus_es --embed-model nomic-embed-text --batch 16")


if __name__ == "__main__":
    main()
