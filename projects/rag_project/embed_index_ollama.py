#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed JSONL chunks with Ollama and index in ChromaDB (persistent).

Prereqs:
- Ollama installed and running (default host http://localhost:11434).
  Install: https://ollama.com
  Start service (macOS): ollama serve
  Pull models (examples):
    ollama pull nomic-embed-text
    ollama pull llama3.1:8b-instruct

- Python deps (already added in requirements.txt):
  chromadb, requests, tqdm

Input JSONL (from chunk_jsonl.py), one item per line:
{
  "doc_id": "...",
  "chunk_id": "...",
  "url": "...",
  "title": "...",
  "author": "...",
  "date": "...",
  "source": "...",
  "language": "es",
  "text": "...",
  "chunk_index": N,
  "n_chars": int
}

Usage:
  python projects/rag_project/embed_index_ollama.py \
    --input projects/rag_project/corpus/articles_chunks.jsonl \
    --persist-dir projects/rag_project/vectorstore \
    --collection genz_corpus_es \
    --embed-model nomic-embed-text \
    --batch 32 --sleep 0.0
"""

import argparse
import json
import os
import time
from typing import Dict, Any, Generator, List

import requests
from tqdm import tqdm
import chromadb
from chromadb.api.types import Embeddings, IDs, Metadatas, Documents
from typing import cast


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


def ollama_embed(text: str, model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> List[float]:
    """
    Call Ollama embeddings endpoint for a single text.
    API: POST /api/embeddings  {"model":"nomic-embed-text","input":"..."}
    Returns list[float] embedding.
    """
    url = f"{host}/api/embeddings"
    resp = requests.post(url, json={"model": model, "input": text}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Single input returns {"embedding": [...]}
    emb = data.get("embedding")
    if emb is None:
        # If the server returns "data": [{"embedding":[...]}] (older variants), handle it
        maybe = data.get("data")
        if isinstance(maybe, list) and maybe and "embedding" in maybe[0]:
            emb = maybe[0]["embedding"]
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected embedding payload: {data}")
    # Validate numeric, non-empty vector
    if len(emb) == 0 or not all(isinstance(x, (int, float)) for x in emb):
        raise RuntimeError(f"Invalid embedding vector (len={len(emb) if isinstance(emb, list) else 'n/a'}) from payload: {data}")
    # Cast to float to satisfy Chroma typing expectations
    emb = [float(x) for x in emb]
    return emb


def main():
    ap = argparse.ArgumentParser(description="Embed chunks with Ollama and index in ChromaDB.")
    ap.add_argument("--input", required=True, help="Chunks JSONL path.")
    ap.add_argument("--persist-dir", required=True, help="ChromaDB persistent directory.")
    ap.add_argument("--collection", required=True, help="Chroma collection name.")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model.")
    ap.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL.")
    ap.add_argument("--batch", type=int, default=16, help="Number of docs to add per Chroma batch.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between embedding calls.")
    args = ap.parse_args()

    os.makedirs(args.persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=args.persist_dir)
    # Use cosine distance by default (Chroma default is cosine for new versions)
    collection = client.get_or_create_collection(args.collection, metadata={"hnsw:space": "cosine"})

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []
    added = 0
    expected_dim = None  # ensure all embeddings have consistent dimensionality

    # To support resume, keep a set of existing ids
    existing_ids = set()
    try:
        # This can be expensive for very large collections; acceptable for mid-size corpora
        existing = collection.peek(limit=100000)
        if existing and "ids" in existing:
            existing_ids = set(existing["ids"])
    except Exception:
        existing_ids = set()

    for rec in tqdm(load_jsonl(args.input), desc="Embedding"):
        chunk_id = rec.get("chunk_id") or rec.get("id")
        text = rec.get("text") or ""
        if not chunk_id or not text.strip():
            continue
        if chunk_id in existing_ids:
            continue

        try:
            emb = ollama_embed(text, model=args.embed_model, host=args.ollama_host)
        except Exception as e:
            print(f"[WARN] Embedding failed for {chunk_id}: {e}")
            continue

        # Guard against empty or inconsistent embedding dimensions
        if not isinstance(emb, list) or len(emb) == 0:
            print(f"[WARN] Empty embedding for {chunk_id}, skipping")
            continue
        if expected_dim is None:
            expected_dim = len(emb)
        elif len(emb) != expected_dim:
            print(f"[WARN] Embedding dim mismatch for {chunk_id}: got {len(emb)} expected {expected_dim}; skipping")
            continue

        ids.append(chunk_id)
        docs.append(text)
        metas.append({
            "doc_id": rec.get("doc_id"),
            "url": rec.get("url"),
            "title": rec.get("title"),
            "author": rec.get("author"),
            "date": rec.get("date"),
            "source": rec.get("source"),
            "language": rec.get("language"),
            "chunk_index": rec.get("chunk_index"),
            "n_chars": rec.get("n_chars"),
        })
        embs.append(emb)

        if len(ids) >= args.batch:
            if ids and embs:
                try:
                    collection.add(
                        ids=cast(IDs, ids),
                        documents=cast(Documents, docs),
                        metadatas=cast(Metadatas, metas),
                        embeddings=cast(Embeddings, embs),
                    )
                    added += len(ids)
                except Exception as e:
                    print(f"[WARN] Chroma add failed for batch starting with {ids[0]}: {e}")
                finally:
                    ids, docs, metas, embs = [], [], [], []

        if args.sleep > 0:
            time.sleep(args.sleep)

    if ids and embs:
        try:
            collection.add(
                ids=cast(IDs, ids),
                documents=cast(Documents, docs),
                metadatas=cast(Metadatas, metas),
                embeddings=cast(Embeddings, embs),
            )
            added += len(ids)
        except Exception as e:
            print(f"[WARN] Chroma add failed for final batch starting with {ids[0]}: {e}")

    print(f"[DONE] Added {added} chunks to collection '{args.collection}' at '{args.persist_dir}'")


if __name__ == "__main__":
    main()
