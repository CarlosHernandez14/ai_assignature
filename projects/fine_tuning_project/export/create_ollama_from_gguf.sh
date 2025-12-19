#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GGUF_PATH="${SCRIPT_DIR}/tutor_gguf/tutor_mejorado.gguf"
MODELFILE_PATH="${SCRIPT_DIR}/Modelfile.lora"
MODEL_NAME="tutor_algoritmos_mejorado"

if [[ ! -f "${GGUF_PATH}" ]]; then
  echo "ERROR: No se encontró el GGUF en: ${GGUF_PATH}"
  echo "Primero ejecuta la conversión a GGUF."
  exit 1
fi

if [[ ! -f "${MODELFILE_PATH}" ]]; then
  echo "ERROR: No se encontró el Modelfile.lora en: ${MODELFILE_PATH}"
  exit 1
fi

echo "Creando modelo '${MODEL_NAME}' en Ollama desde GGUF..."
ollama create "${MODEL_NAME}" --file "${MODELFILE_PATH}"

echo "Modelo creado: ${MODEL_NAME}"
echo "Ejecuta:  ollama run ${MODEL_NAME}"
