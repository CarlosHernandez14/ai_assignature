#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download YouTube comments from a list of videos and save them to JSONL for RAG.

Prereqs:
  pip install youtube-comment-downloader

Input:
  --urls-file: text file with one YouTube URL per line (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
  or pass URLs directly via --urls.

Output JSONL schema (one comment per line):
{
  "video_url": "...",
  "video_id": "...",
  "comment_id": "...",
  "author": "...",
  "text": "...",
  "time": "ISO-8601 when available or raw string",
  "like_count": int,
  "reply_count": int,
  "scraped_at": "UTC iso time",
  "id": "video_id|comment_id|timestamp"
}

Usage:
  python projects/rag_project/youtube_comments_scraper.py \
    --urls-file https://www.youtube.com/watch\?v\=vm5tGIDUS9E \
    --out projects/rag_project/corpus/youtube_comments.jsonl \
    --max-comments 500
"""

import argparse
import json
import os
from datetime import datetime
from typing import Iterable, List, Optional
from urllib.parse import urlparse, parse_qs

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT


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


def get_video_id(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        if parsed.hostname in ("www.youtube.com", "youtube.com"):
            qs = parse_qs(parsed.query)
            vid = qs.get("v", [None])[0]
            return vid
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")
    except Exception:
        return None
    return None


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    ensure_outdir(path)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def scrape_comments(video_url: str, sort: int, max_comments: int) -> List[dict]:
    downloader = YoutubeCommentDownloader()
    # sort can be SORT_BY_POPULAR or SORT_BY_RECENT
    gen = downloader.get_comments_from_url(video_url, sort_by=sort)
    out: List[dict] = []
    scraped_at = datetime.utcnow().isoformat() + "Z"
    video_id = get_video_id(video_url) or ""

    for idx, c in enumerate(gen):
        if idx >= max_comments:
            break
        # Library returns fields like "text", "time", "author", "votes", "commentId", "replyCount"
        rec = {
            "video_url": video_url,
            "video_id": video_id,
            "comment_id": c.get("commentId"),
            "author": c.get("author"),
            "text": c.get("text"),
            "time": c.get("time"),
            "like_count": c.get("votes"),
            "reply_count": c.get("replyCount"),
            "scraped_at": scraped_at,
        }
        rec["id"] = f'{video_id}|{rec.get("comment_id","")}|{scraped_at}'
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description="Scrape YouTube comments to JSONL for RAG.")
    ap.add_argument("--urls-file", type=str, default=None, help="Path to file with YouTube URLs (one per line).")
    ap.add_argument("--urls", nargs="*", default=None, help="YouTube URLs.")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL file path.")
    ap.add_argument("--max-comments", type=int, default=300, help="Max comments per video.")
    ap.add_argument("--sort", type=str, default="recent", choices=["recent", "popular"], help="Sorting of comments.")
    args = ap.parse_args()

    urls: List[str] = []
    if args.urls:
        urls.extend(args.urls)
    if args.urls_file:
        urls.extend(read_lines_file(args.urls_file))
    urls = [u for u in urls if u]

    if not urls:
        print("[ERROR] No URLs provided. Use --urls-file or --urls.")
        return

    sort = SORT_BY_RECENT if args.sort == "recent" else SORT_BY_POPULAR

    total = 0
    for u in urls:
        try:
            recs = scrape_comments(u, sort=sort, max_comments=args.max_comments)
            total += write_jsonl(args.out, recs)
            print(f"[OK] {u} -> {len(recs)} comments")
        except Exception as e:
            print(f"[WARN] Failed {u}: {e}")

    print(f"[DONE] Wrote {total} comments to {args.out}")


if __name__ == "__main__":
    main()
