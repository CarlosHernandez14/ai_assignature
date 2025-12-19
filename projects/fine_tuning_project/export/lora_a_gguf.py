from __future__ import annotations

import argparse
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------------
# Utilidades de sistema
# -------------------------
def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(code)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


# -------------------------
# Gestión de llama.cpp
# -------------------------
def ensure_llama_cpp_exists(llama_cpp_dir: Path) -> None:
    """
    Asegura que llama.cpp esté disponible, clonándolo si no existe.
    Requiere 'git' si se necesita clonar.
    """
    if llama_cpp_dir.exists():
        info(f"Usando llama.cpp de: {llama_cpp_dir}")
        return
    git = which("git")
    if not git:
        fail("No se encontró 'git' en PATH y llama.cpp no existe. Instala git o provee --llama-cpp-dir existente.")
    repo = "https://github.com/ggerganov/llama.cpp.git"
    info(f"Clonando llama.cpp en {llama_cpp_dir} ...")
    subprocess.run([git, "clone", "--depth", "1", repo, str(llama_cpp_dir)], check=True)
    info("llama.cpp clonado.")


def find_convert_script(llama_cpp_dir: Path) -> Path:
    """
    Localiza el script de conversión a GGUF, considerando variaciones de nombre/ubicación.
    """
    candidates: List[Path] = [
        llama_cpp_dir / "convert-hf-to-gguf.py",
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "scripts" / "convert-hf-to-gguf.py",
        llama_cpp_dir / "scripts" / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert_hf_to_gguf_update.py",
        llama_cpp_dir / "scripts" / "convert_hf_to_gguf_update.py",
    ]
    for c in candidates:
        if c.exists():
            info(f"Script de conversión encontrado: {c}")
            return c
    fail(f"No se encontró el script convert*_to_gguf.py en {llama_cpp_dir}")


# -------------------------
# Validaciones de entrada
# -------------------------
def validate_inputs(base: str, lora_dir: Path, out_path: Path) -> None:
    if not lora_dir.exists():
        fail(f"No existe el directorio LoRA: {lora_dir}")
    adapter_cfg = lora_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        fail(f"No se encontró adapter_config.json en {lora_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Base puede ser un id de HF Hub o una ruta local; lo valida transformers en load.


# -------------------------
# Conversión / Fusión
# -------------------------
def merge_lora_to_tmp(base: str, lora_dir: Path, dtype: torch.dtype) -> Path:
    """
    Fusiona el LoRA con el modelo base y guarda un modelo HF temporal.
    Retorna la ruta al directorio temporal con el modelo fusionado.
    """
    info("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)

    info("Cargando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)

    info("Cargando LoRA...")
    model = PeftModel.from_pretrained(base_model, str(lora_dir))

    info("Fusionando LoRA con el modelo base (merge_and_unload)...")
    merged = model.merge_and_unload()

    tmp_dir = Path(tempfile.mkdtemp(prefix="merged_hf_"))
    info(f"Guardando modelo fusionado temporal en: {tmp_dir}")
    merged.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)
    return tmp_dir


def run_convert_to_gguf(convert_script: Path, merged_dir: Path, out_path: Path, outtype: str) -> None:
    """
    Ejecuta el script de llama.cpp para convertir HF->GGUF.
    outtype: 'f16' o 'f32'
    """
    info("Convirtiendo a GGUF...")
    # Usamos el mismo intérprete actual de Python
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        outtype,
    ]
    subprocess.run(cmd, check=True)
    info(f"GGUF escrito en: {out_path}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Fusiona LoRA + base y convierte a GGUF (para Ollama) usando llama.cpp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base", required=True, help="Modelo base HF (nombre o ruta local)")
    ap.add_argument("--lora", required=True, help="Ruta del LoRA entrenado (directorio con adapter_config.json)")
    ap.add_argument("--out", required=True, help="Ruta de salida del GGUF, p.ej. export/tutor_gguf/tutor_f16.gguf")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Precisión al fusionar antes de GGUF")
    return ap.parse_args()


def main():
    args = parse_args()

    lora_dir = Path(args.lora).resolve()
    out_path = Path(args.out).resolve()

    validate_inputs(args.base, lora_dir, out_path)

    # Selección de dtype e 'outtype' para GGUF
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    outtype = "f16" if args.dtype == "float16" else "f32"

    # Fusionar LoRA -> modelo HF temporal
    try:
        merged_dir = merge_lora_to_tmp(args.base, lora_dir, torch_dtype)
    except Exception as e:
        fail(f"Fallo fusionando LoRA+base: {e}")

    # Preparar llama.cpp
    script_dir = Path(__file__).resolve().parent
    if args.llama_cpp_dir:
        llama_cpp_dir = Path(args.llama_cpp_dir).resolve()
    else:
        llama_cpp_dir = script_dir / "llama.cpp"

    try:
        ensure_llama_cpp_exists(llama_cpp_dir)
    except subprocess.CalledProcessError as e:
        fail(f"No se pudo clonar llama.cpp automáticamente: {e}")
    except Exception as e:
        fail(f"Error preparando llama.cpp: {e}")

    convert_script = find_convert_script(llama_cpp_dir)

    # Ejecutar conversión a GGUF
    try:
        run_convert_to_gguf(convert_script, merged_dir, out_path, outtype)
    except subprocess.CalledProcessError as e:
        fail(f"Fallo ejecutando convert-hf-to-gguf: {e}")
    finally:
        # Limpiar directorio temporal
        try:
            shutil.rmtree(merged_dir, ignore_errors=True)
            info(f"Limpiado temporal: {merged_dir}")
        except Exception:
            pass

    print("✅ Conversión completada correctamente.")


if __name__ == "__main__":
    main()
