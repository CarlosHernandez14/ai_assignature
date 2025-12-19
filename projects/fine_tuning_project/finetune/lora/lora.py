
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# -------------------------
# Utilidades de dispositivo
# -------------------------
def detect_device_and_dtype() -> Tuple[str, torch.dtype]:
    """
    Selecciona dispositivo y dtype adecuados para CPU/MPS.
    - En MPS (Apple Silicon) usar float32 para estabilidad de backprop.
    - En CPU también se usa float32.
    """
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    dtype = torch.float32
    return device, dtype


# ------------------------------------
# Sugerencia de módulos objetivo de LoRA
# ------------------------------------
def suggest_target_modules(model) -> List[str]:
    """
    Intenta sugerir target_modules compatibles con distintas arquitecturas.
    - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Phi-3: qkv_proj, o_proj, gate_up_proj, down_proj
    Si no se detecta, usa un fallback razonable.
    """
    name_set = {n for n, _ in model.named_modules()}
    candidates = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    ]
    for cand in candidates:
        if any(any(k in n for n in name_set) for k in cand):
            return cand
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


# -----------------------
# Validaciones de JSONL
# -----------------------
def validate_jsonl(path: str, sample: int = 3) -> None:
    """
    Realiza una validación rápida del archivo JSONL.
    Levanta ValueError si encuentra líneas corruptas en las primeras 'sample'.
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > sample:
                break
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Línea no es un objeto JSON.")
                _ = (obj.get("instruction") or "").strip()
                _ = (obj.get("response") or "").strip()
            except Exception as e:
                raise ValueError(f"JSONL inválido en {path}, línea {i}: {e}") from e


# -----------------------
# Plantilla de prompts
# -----------------------
def build_prompt(instruction: str) -> str:
    """
    Debe ser consistente entre entrenamiento e inferencia.
    """
    return f"Instrucción: {instruction}\nRespuesta: "


# -----------------------
# Config CLI (dataclass)
# -----------------------
@dataclass
class TrainConfig:
    base: str
    train_path: str
    val_path: str
    out_dir: str
    epochs: int
    lr: float
    batch_size: int
    grad_accum: int
    max_seq_len: int
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    weight_decay: float
    warmup_ratio: float
    eval_steps: int
    save_steps: int
    logging_steps: int
    gradient_checkpointing: bool


# -----------------------
# Preparación de modelo
# -----------------------
def prepare_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def prepare_model(base: str, dtype: torch.dtype, device: str, cfg: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    model.config.use_cache = False  # necesario para grad checkpointing/entrenamiento estable

    # Mover a dispositivo
    model.to(torch.device(device))

    # Configurar LoRA
    target_modules = suggest_target_modules(model)
    print(f"[INFO] target_modules LoRA: {target_modules}")
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("[INFO] Gradient checkpointing habilitado.")
        except Exception as e:
            print(f"[WARN] No se pudo habilitar gradient checkpointing: {e}")

    return model


# -----------------------
# Preprocesamiento
# -----------------------
def make_preprocess(tokenizer, max_seq_len: int):
    eos = tokenizer.eos_token or "</s>"

    def preprocess(example: Dict) -> Dict:
        instr = (example.get("instruction") or "").strip()
        resp = (example.get("response") or "").strip()
        prompt_text = build_prompt(instr)
        full_text = prompt_text + resp + eos

        # Tokenizar por separado para enmascarar correctamente el prompt
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"]

        # Enmascarar los tokens del prompt
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return preprocess


# -----------------------
# CLI / Main
# -----------------------
def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(
        description="Entrenamiento LoRA para tutor de algoritmos (CPU/MPS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Modelo base HF")
    ap.add_argument("--train", dest="train_path", type=str, required=True, help="Ruta a train.jsonl")
    ap.add_argument("--val", dest="val_path", type=str, required=True, help="Ruta a val.jsonl")
    ap.add_argument("--out-dir", type=str, required=True, help="Directorio de salida de los adaptadores LoRA")

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--gradient-checkpointing", action="store_true", help="Habilita gradient checkpointing (si es compatible)")

    args = ap.parse_args()

    return TrainConfig(
        base=args.base,
        train_path=args.train_path,
        val_path=args.val_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Validar datasets rapidamente
    validate_jsonl(cfg.train_path)
    validate_jsonl(cfg.val_path)

    # Dispositivo
    device, dtype = detect_device_and_dtype()
    print(f"[INFO] Dispositivo: {device} | dtype: {dtype}")

    # Tokenizador / Modelo
    print("[INFO] Cargando tokenizer...")
    tokenizer = prepare_tokenizer(cfg.base)

    print("[INFO] Cargando modelo base y aplicando LoRA...")
    model = prepare_model(cfg.base, dtype, device, cfg)

    # Dataset
    dataset = load_dataset(
        "json",
        data_files={"train": cfg.train_path, "validation": cfg.val_path},
    )

    preprocess = make_preprocess(tokenizer, cfg.max_seq_len)
    tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

    # Collator (causal LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Entrenamiento
    print("[INFO] Configurando entrenamiento...")
    # Compatibilidad con distintas versiones de Transformers:
    # Filtramos kwargs por la firma de TrainingArguments y aplicamos alias cuando cambian nombres.
    import inspect
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())

    base_kwargs = {
        "output_dir": cfg.out_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": cfg.grad_accum,
        "logging_steps": cfg.logging_steps,
        "num_train_epochs": cfg.epochs,
        "learning_rate": cfg.lr,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "eval_steps": cfg.eval_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "seed": cfg.seed,
        "dataloader_num_workers": 0,      # seguro para macOS
        "dataloader_pin_memory": False,   # evitar warning en MPS
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "optim": "adamw_torch",           # evitar fused optimizer en CPU/MPS
        "report_to": [],                  # sin trackers externos
    }

    # Estrategias de evaluación/guardado con nombres compatibles
    if "evaluation_strategy" in ta_params:
        base_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        base_kwargs["eval_strategy"] = "steps"

    if "save_strategy" in ta_params:
        base_kwargs["save_strategy"] = "steps"

    # MPS solo si el argumento existe en esta versión
    if device == "mps" and "use_mps_device" in ta_params:
        base_kwargs["use_mps_device"] = True

    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**filtered_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("[INFO] Iniciando entrenamiento...")
    trainer.train()

    print("[INFO] Guardando adaptadores LoRA en", cfg.out_dir)
    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    print("[DONE] Entrenamiento completado.")


if __name__ == "__main__":
    main()
