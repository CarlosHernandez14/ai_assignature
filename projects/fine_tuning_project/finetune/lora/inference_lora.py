
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------------
# Utilidades de dispositivo
# -------------------------
def detect_device_and_dtype() -> Tuple[str, torch.dtype]:
    """
    Selecciona el mejor backend disponible para inferencia.
    - En MPS (Apple Silicon) se usa float16 para mayor velocidad.
    - En CPU se usa float32.
    """
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    dtype = torch.float16 if use_mps else torch.float32
    return device, dtype


# -------------------------
# Plantilla de prompts
# -------------------------
def build_prompt(instruction: str) -> str:
    """
    Debe coincidir con el formato usado en entrenamiento.
    """
    return f"Instrucción: {instruction}\nRespuesta: "


def extract_response(full_text: str) -> str:
    """
    Extrae solo la parte posterior a 'Respuesta: ' si está presente.
    """
    split_token = "\nRespuesta: "
    if split_token in full_text:
        return full_text.split(split_token, 1)[1].strip()
    return full_text.strip()


# -------------------------
# Configuración CLI
# -------------------------
@dataclass
class InferConfig:
    base: str
    lora: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    merge_weights: bool


def parse_args() -> InferConfig:
    ap = argparse.ArgumentParser(
        description="Inferencia con LoRA para tutor de algoritmos (CPU/MPS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base", required=True, help="Modelo base HF (ej. microsoft/Phi-3-mini-4k-instruct)")
    ap.add_argument("--lora", required=True, help="Ruta a carpeta de adaptadores LoRA (output de train)")
    ap.add_argument("--prompt", required=True, help="Instrucción del usuario")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--merge-weights", action="store_true", help="Fusiona LoRA con base antes de generar")
    args = ap.parse_args()

    return InferConfig(
        base=args.base,
        lora=args.lora,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        merge_weights=args.merge_weights,
    )


# -------------------------
# Main
# -------------------------
def main():
    cfg = parse_args()

    device, dtype = detect_device_and_dtype()
    print(f"[INFO] Dispositivo: {device}, dtype: {dtype}")

    print("[INFO] Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Cargando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base, torch_dtype=dtype)
    if device == "mps":
        base_model.to("mps")

    print("[INFO] Cargando adaptadores LoRA...")
    model = PeftModel.from_pretrained(base_model, cfg.lora)

    if cfg.merge_weights:
        # Funde los pesos LoRA con el modelo base para acelerar inferencia
        print("[INFO] Fusionando LoRA con el modelo base (merge_and_unload)...")
        model = model.merge_and_unload()
        if device == "mps":
            model.to("mps")

    user_prompt = build_prompt(cfg.prompt)
    inputs = tokenizer(user_prompt, return_tensors="pt")
    if device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    print("[INFO] Generando...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = extract_response(text)

    print("\n=== Respuesta del Tutor ===\n")
    print(response)


if __name__ == "__main__":
    main()
