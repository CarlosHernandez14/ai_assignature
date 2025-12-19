
import argparse
import json
import os
from datetime import datetime
from typing import Iterable, List, Optional
from pathlib import Path

from pypdf import PdfReader


def ensure_outdir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    ensure_outdir(path)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def pdfs_in_dir(root: str) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                out.append(os.path.join(base, fn))
    return out


def clean_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s or None


def extract_pdf_whole(pdf_path: str, min_chars: int) -> Optional[dict]:
    try:
        reader = PdfReader(pdf_path)
    except Exception:
        return None

    texts: List[str] = []
    for i in range(len(reader.pages)):
        try:
            page = reader.pages[i]
            txt = page.extract_text() or ""
            txt = txt.strip()
            if txt:
                texts.append(txt)
        except Exception:
            # skip unreadable page
            continue

    joined = "\n\n".join(texts).strip()
    if len(joined) < min_chars:
        return None

    meta = reader.metadata or {}
    title = clean_str(getattr(meta, "title", None) or meta.get("/Title") if isinstance(meta, dict) else None)
    author = clean_str(getattr(meta, "author", None) or meta.get("/Author") if isinstance(meta, dict) else None)
    date = clean_str(getattr(meta, "creation_date", None) or meta.get("/CreationDate") if isinstance(meta, dict) else None)

    url = Path(pdf_path).resolve().as_uri()
    scraped_at = datetime.utcnow().isoformat() + "Z"
    return {
        "url": url,
        "title": title,
        "author": author,
        "date": date,
        "text": joined,
        "source": "local-pdf",
        "language": "es",
        "scraped_at": scraped_at,
        "id": f"{url}|{scraped_at}",
    }


def extract_pdf_pages(pdf_path: str, min_chars: int) -> List[dict]:
    try:
        reader = PdfReader(pdf_path)
    except Exception:
        return []

    meta = reader.metadata or {}
    title = clean_str(getattr(meta, "title", None) or meta.get("/Title") if isinstance(meta, dict) else None)
    author = clean_str(getattr(meta, "author", None) or meta.get("/Author") if isinstance(meta, dict) else None)
    date = clean_str(getattr(meta, "creation_date", None) or meta.get("/CreationDate") if isinstance(meta, dict) else None)

    url_base = Path(pdf_path).resolve().as_uri()
    scraped_at = datetime.utcnow().isoformat() + "Z"
    n_pages = len(reader.pages)
    out: List[dict] = []

    for i in range(n_pages):
        try:
            page_obj = reader.pages[i]
            txt = (page_obj.extract_text() or "").strip()
            if len(txt) < min_chars:
                continue
            url_page = f"{url_base}#page={i+1}"
            out.append({
                "url": url_page,
                "title": title,
                "author": author,
                "date": date,
                "text": txt,
                "page": i + 1,
                "n_pages": n_pages,
                "source": "local-pdf",
                "language": "es",
                "scraped_at": scraped_at,
                "id": f"{url_page}|{scraped_at}",
            })
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description="Ingest local PDFs into JSONL for RAG.")
    ap.add_argument("--input-dir", required=True, help="Directory containing PDFs (scans not supported unless OCRed).")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--per-page", action="store_true", help="Emit one JSON per PDF page instead of one per document.")
    ap.add_argument("--min-chars", type=int, default=200, help="Skip items shorter than this many chars.")
    args = ap.parse_args()

    paths = pdfs_in_dir(args.input_dir)
    print(f"[INFO] Found {len(paths)} PDFs under {args.input_dir}")
    total = 0

    for p in paths:
        try:
            if args.per_page:
                recs = extract_pdf_pages(p, min_chars=args.min_chars)
                total += write_jsonl(args.out, recs)
                print(f"[OK] {p} -> {len(recs)} pages")
            else:
                rec = extract_pdf_whole(p, min_chars=args.min_chars)
                if rec:
                    total += write_jsonl(args.out, [rec])
                    print(f"[OK] {p}")
                else:
                    print(f"[WARN] Skipped (too short/unreadable): {p}")
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}")

    print(f"[DONE] Wrote {total} records to {args.out}")


if __name__ == "__main__":
    main()
