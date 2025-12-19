
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import sys
# Ensure local imports work when running as a script
_local_dir = Path(__file__).resolve().parent
if str(_local_dir) not in sys.path:
    sys.path.insert(0, str(_local_dir))
from chroma_vector_embedings import retriever as default_retriever, get_retriever


# ---------- Paths & defaults ----------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "preguntas.txt"
DEFAULT_OUTPUT = BASE_DIR / "INFORME_FINAL_COMPLETO.md"

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral-nemo:latest")
DEFAULT_K = int(os.getenv("RAG_K", "10"))
DEFAULT_CHUNK_CLIP = int(os.getenv("RAG_CHUNK_CLIP", "800"))


# ---------- Prompt ----------

PROMPT_TMPL = """
Eres un investigador experto en filosofía y análisis de datos. Proyecto: "La Generación Z y la Crisis de Sentido".

OBJETIVO: Responde la siguiente pregunta de investigación sintetizando:
1. TEORÍA: Conceptos filosóficos (Heidegger, Han, Bauman, etc.) presentes en el contexto.
2. EVIDENCIA: Datos empíricos (YouTube, encuestas, artículos) presentes en el contexto.

CONTEXTO RECUPERADO:
{context}

PREGUNTA DE INVESTIGACIÓN:
{question}

INSTRUCCIONES DE RESPUESTA:
- Escribe un análisis profundo y estructurado (mínimo 2 párrafos).
- Cita las fuentes teóricas y empíricas explícitamente.
- Si hay contradicciones entre la teoría y los datos, señálalas.
- Responde en español académico.

ANÁLISIS:
""".strip()


def build_model(model_name: str) -> ChatOllama:
    # Extra kwargs can be exposed via env/CLI (e.g., temperature/top_p) if desired
    return ChatOllama(model=model_name)


def read_questions(path: Path) -> List[str]:
    if not path.exists():
        print(f"❌ Error: No se encontró el archivo '{path}'")
        return []
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def format_context(docs: Iterable, clip_chars: int = DEFAULT_CHUNK_CLIP) -> str:
    """
    Build a readable context section with per-chunk metadata and clipped content.
    """
    parts: List[str] = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        src = str(md.get("source", "desconocido")).upper()
        title = md.get("title")
        url = md.get("url")
        author = md.get("author")
        head: List[str] = [src]
        if title:
            head.append(f"título={title}")
        if author:
            head.append(f"autor={author}")
        if url:
            head.append(f"url={url}")
        header = "[" + " | ".join(head) + "]"
        content = (d.page_content or "").replace("\n", " ").strip()
        if clip_chars > 0 and len(content) > clip_chars:
            content = content[:clip_chars].rstrip() + " ..."
        parts.append(f"{header}\n{content}")
    if not parts:
        return "(No se recuperó contexto relevante para esta pregunta.)"
    return "\n\n".join(parts)


def write_header(fh, model_name: str, total: int) -> None:
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    fh.write("# Informe de Investigación: Crisis de Sentido Gen Z\n")
    fh.write(f"**Fecha de generación:** {timestamp}\n")
    fh.write(f"**Modelo:** {model_name} + RAG\n")
    fh.write(f"**Total de preguntas:** {total}\n")
    fh.write("---\n\n")


def run(input_file: Path, output_file: Path, model_name: str, k: int, clip_chars: int) -> None:
    print("\n=== CORRIENDO ANALISIS DEL MODELO ===")

    questions = read_questions(input_file)
    if not questions:
        return

    print(f"📂 Se encontraron {len(questions)} preguntas en '{input_file}'.")
    print(f"📝 El resultado se escribirá en '{output_file}'\n")

    model = build_model(model_name)
    prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)
    chain = prompt | model

    # Use a retriever configured for desired k
    rtv = get_retriever(k=k, search_type="mmr") if k != DEFAULT_K else default_retriever

    # Prepare output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        write_header(fh, model_name=model_name, total=len(questions))

    # Process each question
    for i, q in enumerate(questions, 1):
        print(f"Procesando pregunta {i}/{len(questions)}: {q[:70]}...")
        t0 = time.time()

        docs = rtv.invoke(q)
        context_text = format_context(docs, clip_chars=clip_chars)
        response = chain.invoke({"context": context_text, "question": q})

        with output_file.open("a", encoding="utf-8") as fh:
            fh.write(f"## {i}. {q}\n\n")
            fh.write(f"{response}\n\n")
            fh.write("**Fuentes consultadas (tipos):**\n")
            # Unique source types
            srcs = sorted({str(getattr(d, 'metadata', {}).get('source', 'desconocido')).upper() for d in docs})
            for s in srcs:
                fh.write(f"- {s}\n")
            fh.write("\n---\n\n")

        dt = time.time() - t0

    print(f"\n🎉 ¡PROCESO COMPLETADO! Revisa el archivo: {output_file}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generador de informe RAG sobre la base vectorial local.")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Archivo de preguntas (una por línea).")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Archivo de salida Markdown.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo de Ollama a usar.")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Número de pasajes a recuperar por pregunta.")
    ap.add_argument("--clip", type=int, default=DEFAULT_CHUNK_CLIP, help="Máx. caracteres por pasaje en el prompt.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    run(
        input_file=input_file,
        output_file=output_file,
        model_name=args.model,
        k=args.k,
        clip_chars=args.clip,
    )


if __name__ == "__main__":
    main()
