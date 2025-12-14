#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RAG query runner:
- Retrieves top-k chunks from ChromaDB by cosine similarity
- Calls Ollama chat model with a Spanish prompt template
- Returns an answer with citations to sources and URLs

Prereqs:
- Run embed_index_ollama.py beforehand to populate the Chroma collection.
- Ollama running locally (http://localhost:11434) with an instruction model pulled, e.g.:
    ollama pull llama3.1:8b-instruct
- chromadb and requests installed.

Usage:
  python projects/rag_project/rag_query_ollama.py \
    --persist-dir projects/rag_project/vectorstore \
    --collection genz_corpus_es \
    --query "¿Qué evidencias empíricas en el corpus muestran una crisis de sentido en la Gen Z?" \
    --llm-model llama3.1:8b-instruct \
    --k 5
"""

import argparse
import json
from typing import List, Dict, Any
import requests
import chromadb


PROMPT_TEMPLATE = """Eres un analista filosófico y socio-tecnológico. Responde en español de forma rigurosa, clara y con citas.
Contexto recuperado (fragmentos):
{context_blocks}

Instrucciones:
- Razona con los marcos: Existencialismo (Camus, Sartre), Posmodernidad (Lyotard), Identidad líquida (Bauman),
  Cultura del rendimiento (Byung-Chul Han), Vigilancia/Biopoder (Foucault), Técnica (Heidegger), Esfera pública (Habermas).
- Incluye citas entre corchetes con [source: dominio o título, url] para cada afirmación clave.
- Si la evidencia es débil, dilo explícitamente. No inventes citas.

Pregunta del usuario:
{question}

Responde:
"""

def call_ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    url = f"{host}/api/generate"
    # Using generate (single-shot) keeps it simple; for multi-turn use /api/chat
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def build_context_block(hits: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        meta = {}
        text = ""
        if isinstance(h, dict):
            meta = h.get("metadatas") or {}
            if not isinstance(meta, dict):
                meta = {}
            text = h.get("documents")
            if isinstance(text, list):
                text = text[0]
            if not isinstance(text, str):
                text = ""
        title = meta.get("title") or ""
        source = meta.get("source") or ""
        url = meta.get("url") or ""
        blocks.append(f"({i}) [{title} — {source}] {url}\n{text}\n")
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser(description="Query RAG over Chroma with Ollama.")
    ap.add_argument("--persist-dir", required=True, help="Chroma persistent dir.")
    ap.add_argument("--collection", required=True, help="Chroma collection name.")
    ap.add_argument("--query", required=True, help="User question in Spanish.")
    ap.add_argument("--k", type=int, default=5, help="Top-k results for retrieval.")
    ap.add_argument("--llm-model", default="llama3.1:8b-instruct", help="Ollama instruction model.")
    ap.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host.")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    coll = client.get_or_create_collection(args.collection, metadata={"hnsw:space": "cosine"})

    # Retrieve
    res = coll.query(
        query_texts=[args.query],
        n_results=args.k,
        include=["documents", "metadatas", "distances"],
    )
    hits: List[Dict[str, Any]] = []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    dists = res.get("distances") or []
    ids = res.get("ids") or []
    if docs:
        n = len(docs[0])
        for i in range(n):
            hits.append({
                "documents": docs[0][i],
                "metadatas": metas[0][i] if metas else {},
                "ids": ids[0][i] if ids else None,
                "distance": dists[0][i] if dists else None,
            })

    if not hits:
        print("[WARN] No retrieval hits. Answering without context.")
        prompt = PROMPT_TEMPLATE.format(context_blocks="(vacío)", question=args.query)
    else:
        ctx = build_context_block(hits)
        prompt = PROMPT_TEMPLATE.format(context_blocks=ctx, question=args.query)

    answer = call_ollama_chat(args.llm_model, prompt, host=args.ollama_host)
    print(answer)


if __name__ == "__main__":
    main()
