
from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple


def _normalize_unicode(s: str) -> str:
    """Aplica normalización Unicode para evitar discrepancias por acentos/formas."""
    return unicodedata.normalize("NFC", s or "")


def normalize_text(s: str) -> str:
    """
    Normaliza texto:
    - Unicode NFC
    - Normaliza saltos de línea a '\n'
    - Colapsa espacios/tabs múltiples
    - Limita saltos consecutivos a máximo 2
    - Recorta espacios al inicio/fin
    """
    s = _normalize_unicode(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def read_jsonl(path: str) -> List[Dict]:
    """Lee JSONL de forma tolerante a errores, saltando líneas inválidas."""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                # Línea corrupta -> descartar
                continue
    return rows


def keep_example(ex: Dict, min_chars: int, max_chars: int) -> bool:
    """Valida que el ejemplo tenga instrucción y respuesta, y longitud dentro de rango."""
    instr = (ex.get("instruction") or "").strip()
    resp = (ex.get("response") or "").strip()
    if not instr or not resp:
        return False
    total_len = len(instr) + 1 + len(resp)
    if total_len < min_chars or total_len > max_chars:
        return False
    return True


def unique_examples(rows: Iterable[Dict]) -> List[Dict]:
    """
    Elimina duplicados exactos basados en la combinación normalizada (instr, resp).
    Conserva el primer ejemplar.
    """
    seen = set()
    out: List[Dict] = []
    for ex in rows:
        key = (
            normalize_text(ex.get("instruction", "")),
            normalize_text(ex.get("response", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append({"instruction": key[0], "response": key[1]})
    return out


def normalize_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    """
    Si las proporciones no suman 1.0 (por redondeos/errores), las reescala.
    """
    s = train + val + test
    if s <= 0:
        raise ValueError("Las proporciones deben ser positivas.")
    if abs(s - 1.0) < 1e-6:
        return train, val, test
    return train / s, val / s, test / s


def split_dataset(data: List[Dict], ratios: Tuple[float, float, float], seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Baraja y divide determinísticamente el dataset según las proporciones indicadas."""
    rng = random.Random(seed)
    data_shuffled = list(data)
    rng.shuffle(data_shuffled)

    n = len(data_shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # Ajuste para asegurar suma exacta
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train = data_shuffled[:n_train]
    val = data_shuffled[n_train : n_train + n_val]
    test = data_shuffled[n_train + n_val :]
    return train, val, test


def write_jsonl(path: str, rows: List[Dict]) -> None:
    """Escribe una lista de objetos en JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_ollama_format(rows: List[Dict]) -> List[Dict]:
    """
    Convierte de {'instruction','response'} a {'prompt','response'}.
    (El SYSTEM se define en el Modelfile de Ollama)
    """
    return [{"prompt": normalize_text(r.get("instruction", "")), "response": normalize_text(r.get("response", ""))} for r in rows]


def main():
    ap = argparse.ArgumentParser(
        description="Limpia y divide dataset (instruction/response) y exporta splits + dataset para Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--in", dest="input_path", required=True, help="Ruta al JSONL de entrada (instruction/response)")
    ap.add_argument("--out-dir", required=True, help="Directorio de salida para los splits")
    ap.add_argument("--train-ratio", type=float, default=0.85)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--min-chars", type=int, default=120, help="Longitud mínima total por ejemplo (instr+resp)")
    ap.add_argument("--max-chars", type=int, default=8000, help="Longitud máxima total por ejemplo (instr+resp)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true", help="Eliminar duplicados exactos tras normalización")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_jsonl(args.input_path)
    print(f"[INFO] Leídos {len(rows)} ejemplos del archivo de entrada.")

    # Normalización + filtro por longitud
    cleaned: List[Dict] = []
    dropped_len = 0
    for ex in rows:
        ex_norm = {
            "instruction": normalize_text(ex.get("instruction", "")),
            "response": normalize_text(ex.get("response", "")),
        }
        if keep_example(ex_norm, args.min_chars, args.max_chars):
            cleaned.append(ex_norm)
        else:
            dropped_len += 1
    print(f"[INFO] Tras normalización/filtro: {len(cleaned)} ejemplos (descartados por longitud: {dropped_len}).")

    # Deduplicación opcional
    before_dedup = len(cleaned)
    if args.dedup:
        cleaned = unique_examples(cleaned)
    print(f"[INFO] Duplicados eliminados: {before_dedup - len(cleaned)} | Total tras deduplicar: {len(cleaned)}")

    # Proporciones
    ratios = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    print(f"[INFO] Ratios (train/val/test): {ratios[0]:.3f}/{ratios[1]:.3f}/{ratios[2]:.3f}")

    # Split determinista
    train, val, test = split_dataset(cleaned, ratios, seed=args.seed)
    print(f"[INFO] Splits -> train: {len(train)}, val: {len(val)}, test: {len(test)} (total: {len(cleaned)})")

    # Guardar LoRA-format
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)
    print("[INFO] Guardados splits en formato LoRA (instruction/response).")

    # Guardar dataset unificado para Ollama
    ollama_rows = to_ollama_format(train + val + test)
    write_jsonl(os.path.join(args.out_dir, "ollama_dataset.jsonl"), ollama_rows)
    print("[INFO] Guardado dataset para Ollama: ollama_dataset.jsonl")

    print(f"[DONE] Datos procesados en: {args.out_dir}")


if __name__ == "__main__":
    main()
