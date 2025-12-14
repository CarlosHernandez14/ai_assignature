#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk a JSONL corpus (one article/document per line) into smaller passages for embeddings/RAG.

Input JSONL expected schema (as produced by scrape_corpus.py):
{
  "url": "...",
  "title": "...",
  "author": "...",
  "date": "...",
  "text": "... long article text ...",
  "source": "... domain ...",
  "language": "es",
  "scraped_at": "...",
  "id": "url|timestamp"
}

Output JSONL (one chunk per line), schema:
{
  "doc_id": "original record id",
  "chunk_id": "doc_id#chunkN",
  "url": "...",
  "title": "...",
  "author": "...",
  "date": "...",
  "source": "...",
  "language": "es",
  "text": "... chunk text ...",
  "chunk_index": N,
  "n_chars": int
}

Usage:
  python projects/rag_project/chunk_jsonl.py \
    --input projects/rag_project/corpus/articles.jsonl \
    --output projects/rag_project/corpus/articles_chunks.jsonl \
    --chunk-size 1000 \
    --overlap 200
"""

import argparse
import io
import json
import os
import re
from typing import Generator, List, Dict, Any


def load_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    # collapse excessive whitespace and normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    # Simple paragraph split on blank lines
    text = normalize_ws(text)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def chunk_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Greedy char-based chunking with overlap. Use paragraphs to seed chunks where possible.
    """
    paras = split_paragraphs(text)
    if not paras:
        return []

    chunks: List[str] = []
    buf = io.StringIO()
    current_len = 0

    def flush_buffer():
        nonlocal buf, current_len
        s = buf.getvalue().strip()
        if s:
            chunks.append(s)
        buf = io.StringIO()
        current_len = 0

    for p in paras:
        p_len = len(p)
        if current_len == 0 and p_len >= chunk_size:
            # paragraph itself is too large; hard-split it
            start = 0
            while start < p_len:
                end = min(start + chunk_size, p_len)
                chunks.append(p[start:end].strip())
                start = max(end - overlap, end)
            continue

        if current_len + p_len + 2 <= chunk_size:
            if current_len > 0:
                buf.write("\n\n")
                current_len += 2
            buf.write(p)
            current_len += p_len
        else:
            # flush current
            flush_buffer()
            # if still too big, split paragraph
            if p_len > chunk_size:
                start = 0
                while start < p_len:
                    end = min(start + chunk_size, p_len)
                    chunks.append(p[start:end].strip())
                    start = max(end - overlap, end)
            else:
                buf.write(p)
                current_len = p_len

    flush_buffer()
    return chunks


def process(input_path: str, output_path: str, chunk_size: int, overlap: int, min_chunk_chars: int) -> int:
    # Clear output if exists (to avoid accidental appends in repeated runs)
    if os.path.exists(output_path):
        os.remove(output_path)

    total = 0
    for rec in load_jsonl(input_path):
        doc_id = rec.get("id") or rec.get("url") or f"doc{total}"
        text = rec.get("text") or ""
        if not text.strip():
            continue

        parts = chunk_long_text(text, chunk_size, overlap)
        out_recs: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(parts):
            if len(chunk) < min_chunk_chars:
                continue
            chunk_id = f"{doc_id}#chunk{idx}"
            out_recs.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "url": rec.get("url"),
                "title": rec.get("title"),
                "author": rec.get("author"),
                "date": rec.get("date"),
                "source": rec.get("source"),
                "language": rec.get("language") or "es",
                "text": chunk,
                "chunk_index": idx,
                "n_chars": len(chunk),
            })
        if out_recs:
            write_jsonl(output_path, out_recs)
            total += len(out_recs)
    return total


def main():
    ap = argparse.ArgumentParser(description="Chunk JSONL documents into passages for RAG.")
    ap.add_argument("--input", required=True, help="Input JSONL file (articles).")
    ap.add_argument("--output", required=True, help="Output JSONL file (chunks).")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Target chunk size in characters.")
    ap.add_argument("--overlap", type=int, default=200, help="Overlap in characters between adjacent chunks.")
    ap.add_argument("--min-chunk-chars", type=int, default=200, help="Drop chunks shorter than this.")
    args = ap.parse_args()

    total = process(args.input, args.output, args.chunk_size, args.overlap, args.min_chunk_chars)
    print(f"[DONE] Wrote {total} chunks to {args.output}")


if __name__ == "__main__":
    main()
