
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------- Paths & defaults ----------

BASE_DIR = Path(__file__).resolve().parent                 # .../projects/rag_project
PROJECTS_DIR = BASE_DIR.parent                             # .../projects
REPO_ROOT = PROJECTS_DIR.parent                            # repo root

# Prefer an existing store at repo root (to remain backward compatible),
# otherwise default to a store inside this project folder.
DEFAULT_PERSIST_CANDIDATES: List[Path] = [
    REPO_ROOT / "vectorstore",
    BASE_DIR / "vectorstore",
]

DEFAULT_COLLECTION = "genz_research"
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
DEFAULT_K = int(os.getenv("RAG_K", "10"))

# Default corpus files (index only if creating a new store)
DEFAULT_SOURCES: List[Tuple[Path, str, Sequence[str]]] = [
    (BASE_DIR / "corpus" / "documents.jsonl", "articulos_externos", ("title", "url", "author")),
    (BASE_DIR / "corpus" / "comentarios_youtube.jsonl", "youtube_comentarios", ("author", "video_id", "like_count")),
]

TEXT_KEYS: Tuple[str, ...] = ("text", "content", "body")  # accepted text keys


def pick_persist_dir(cli_persist: Optional[str]) -> Path:
    if cli_persist:
        return Path(cli_persist).expanduser().resolve()
    for p in DEFAULT_PERSIST_CANDIDATES:
        if p.exists():
            return p.resolve()
    # Fallback: create under project folder
    return (BASE_DIR / "vectorstore").resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_text(record: Dict) -> Optional[str]:
    for k in TEXT_KEYS:
        val = record.get(k)
        if isinstance(val, str) and val.strip():
            return val
    return None


def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jsonl_as_documents(path: Path, source_name: str, metadata_keys: Sequence[str]) -> List[Document]:
    docs: List[Document] = []
    if not path.exists():
        print(f"[WARN] JSONL not found: {path}")
        return docs

    print(f"[INFO] Loading: {source_name} from {path}")
    for rec in iter_jsonl(path):
        txt = detect_text(rec)
        if not txt:
            continue
        meta = {"source": source_name}
        for k in metadata_keys:
            if k in rec and rec[k] is not None:
                meta[k] = str(rec[k])
        docs.append(Document(page_content=txt, metadata=meta))
    print(f"[INFO] Loaded {len(docs)} docs from {path.name}")
    return docs


def build_documents_from_sources(sources: Sequence[Tuple[Path, str, Sequence[str]]]) -> List[Document]:
    all_docs: List[Document] = []
    for (path, src_name, meta_keys) in sources:
        all_docs.extend(load_jsonl_as_documents(path, src_name, meta_keys))
    return all_docs


def connect_or_create_store(
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    sources: Optional[Sequence[Tuple[Path, str, Sequence[str]]]] = None,
    recreate: bool = False,
) -> Chroma:
    ensure_dir(persist_dir)

    embeddings = OllamaEmbeddings(model=embed_model)

    # Recreate if requested
    if recreate:
        # Remove old collection directory contents to force a clean build
        # (Chroma will handle internal files; a simple strategy is to delete the directory)
        # Note: To avoid accidental data loss, only do this if explicitly requested.
        import shutil
        print(f"[INFO] Recreating vector store at {persist_dir}")
        shutil.rmtree(persist_dir, ignore_errors=True)
        ensure_dir(persist_dir)

    # If a chroma DB exists in this directory, connecting will reuse it.
    try:
        # Try to connect to existing collection
        store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        # If it has no collections yet AND not recreating, we may still need to seed it
        # We will detect empty by attempting to count; if it errors, we'll rebuild.
        try:
            _ = store._collection.count()  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        # If connection fails, we will build from documents
        store = None  # type: ignore[assignment]

    # Create if empty or after recreation
    if recreate or not (persist_dir / "chroma.sqlite3").exists():
        docs: List[Document] = build_documents_from_sources(sources or DEFAULT_SOURCES)
        if docs:
            print(f"[INFO] Creating new Chroma collection '{collection_name}' with {len(docs)} docs")
            store = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=str(persist_dir),
                collection_name=collection_name,
            )
        else:
            print("[WARN] No documents found to index; creating an empty collection")
            store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name=collection_name,
            )
    else:
        # Ensure we have a usable store instance
        if "store" not in locals() or store is None:
            store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name=collection_name,
            )
        print("[INFO] Existing vector store found. Connected.")

    return store


def get_retriever(k: int = DEFAULT_K, search_type: str = "mmr"):
    """
    Build a retriever with desired strategy.
    search_type: "mmr" (diversity) or "similarity"
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


# ---------- Module-level bootstrap (for `from vector import retriever`) ----------

# Choose a persist dir (prefer existing at repo root)
PERSIST_DIR = pick_persist_dir(cli_persist=None)
vector_store: Chroma = connect_or_create_store(
    persist_dir=PERSIST_DIR,
    collection_name=DEFAULT_COLLECTION,
    embed_model=DEFAULT_EMBED_MODEL,
    sources=DEFAULT_SOURCES,
    recreate=False,
)
retriever = get_retriever(k=DEFAULT_K, search_type="mmr")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build or connect to a Chroma vector store for RAG.")
    ap.add_argument("--persist-dir", type=str, default=None, help="Directory to persist the Chroma DB.")
    ap.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Collection name.")
    ap.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    ap.add_argument("--recreate", action="store_true", help="Recreate the store from default sources.")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Retriever top-k.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    persist_dir = pick_persist_dir(args.persist_dir)

    store = connect_or_create_store(
        persist_dir=persist_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        sources=DEFAULT_SOURCES,
        recreate=args.recreate,
    )

    # Show a quick summary
    try:
        count = store._collection.count()  # type: ignore[attr-defined]
    except Exception:
        count = "unknown"
    print(f"[INFO] Persist dir: {persist_dir}")
    print(f"[INFO] Collection: {args.collection}")
    print(f"[INFO] Docs in collection: {count}")
    # Update module-level globals to reflect CLI run
    global vector_store, retriever
    vector_store = store
    retriever = get_retriever(k=args.k)
    print("[DONE] Vector store ready.")


if __name__ == "__main__":
    main()
