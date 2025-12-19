# Tutor de Algoritmos – Resumen del Fine‑Tuning

Este proyecto entrena un “tutor” de algoritmos que responde con claridad: define conceptos, muestra pasos, hace trazas y comenta complejidad. Todo corre de forma local (CPU o MPS en Mac). Aquí va una explicación corta, sin mucha jerga técnica.

## ¿Qué es el fine‑tuning y por qué LoRA?

- Fine‑tuning: es “ajustar” un modelo de lenguaje ya preentrenado para que hable con el estilo y contenido que yo necesito (en este caso, tutor de algoritmos).
- LoRA: en vez de cambiar todos los parámetros del modelo (lo cual es pesado), se agregan “adaptadores” pequeños que sí se entrenan. Esto hace el proceso más ligero y viable en equipos sin GPU potente.

## Flujo del proyecto

1) Dataset de ejemplo (instrucción → respuesta)
   - Se preparan pares donde la instrucción pide algo (“Explícame Dijkstra con una traza…”) y la respuesta explica con estructura (definición, pasos, ejemplo, complejidad, errores típicos).
   - El proyecto incluye un dataset manual de 500 instrucciones y herramientas para limpiarlo.

2) Limpieza y partición
   - Se normaliza el texto, se quitan duplicados y se divide en entrenamiento, validación y prueba.
   - También se genera un formato alternativo para usar con Ollama si se quiere.

3) Entrenamiento con LoRA
   - Se toma un modelo base pequeño (por ejemplo, Phi‑3 mini) y se le aplican los adaptadores LoRA.
   - Se entrena solo lo necesario para que el modelo adopte el estilo “tutor de algoritmos”.

4) Prueba rápida
   - Con el modelo ajustado, se le hacen preguntas típicas de algoritmos para comprobar que responde ordenado y coherente.

5) (Opcional) Exportar para Ollama
   - Si se desea, se fusiona el LoRA con el modelo base y se convierte a formato GGUF para cargarlo en Ollama y usarlo como un modelo local.

## ¿Qué hay en cada carpeta?

- dataset/
  - Generar y limpiar datos. La idea es obtener pares “instrucción → respuesta” de buena calidad.
- finetune/lora/
  - Entrenamiento con LoRA y una prueba básica de inferencia.
- export/
  - Fusión y conversión a GGUF (para usar el tutor en Ollama de forma local).
- data/
  - raw: datos crudos (ej. el set manual).
  - processed: datos limpios y divididos (train/val/test) listos para entrenar.

## ¿Cómo queda el “tutor” después?

- Responde con una estructura clara: idea principal, pasos, ejemplo simple y análisis de complejidad.
- Suele evitar explicaciones vagas y se enfoca en lo pedagógico (qué hacer, cómo y por qué).
- No es perfecto, pero mejora con más ejemplos bien redactados y algunas épocas extra de entrenamiento.

## Consejos rápidos para mejorar resultados

- Cuidar la calidad del dataset: instrucciones claras y respuestas con buen orden.
- Mezclar tipos de ejercicios: explicaciones, trazas, depuración, comparaciones, elecciones de estructura de datos, etc.
- Mantener variedad y evitar repeticiones exactas (para no sobreajustar).
- Empezar con modelos base pequeños para que el equipo lo soporte bien.

## Requisitos de hardware 

- Funciona en CPU o en Mac con MPS (Apple Silicon).
- No se necesita GPU NVIDIA ni librerías especiales de CUDA.
- LoRA permite entrenar con recursos limitados.

---

En resumen: se prepara un buen dataset de “instrucción → respuesta”, se limpia, se entrena un LoRA sobre un modelo base pequeño, se prueba y, si hace falta, se exporta a Ollama. El resultado es un tutor de algoritmos local y usable para estudiar y practicar.

## Archivos del proyecto

- requirements.txt — Lista de dependencias para que el proyecto funcione en CPU/MPS.
- dataset/clean_and_split.py — Limpia, normaliza y divide el dataset; además genera:
  - Formato LoRA: train/val/test (instruction→response)
  - Formato Ollama: ollama_dataset.jsonl (prompt→response)
- data/raw/manual_500_part1.jsonl y data/raw/manual_500_part2.jsonl — Dataset manual.

- finetune/lora/lora.py — Entrenamiento LoRA sobre un modelo base pequeño (ajusta “adaptadores” ligeros).
- finetune/lora/inference_lora.py — Prueba rápida del LoRA entrenado con una instrucción de ejemplo.
- finetune/lora/outputs/ — Carpeta donde se guardan los adaptadores LoRA resultantes.

- export/lora_a_gguf.py — Funde LoRA + base y convierte a GGUF usando herramientas de llama.cpp.
- export/Modelfile.lora — Plantilla para crear el modelo en Ollama a partir del GGUF exportado.
- export/create_ollama_from_gguf.sh — Script para registrar el modelo en Ollama desde el GGUF.
- export/llama.cpp/ — Herramientas internas para la conversión a GGUF (no hace falta modificarlas).
- export/tutor_gguf/tutor_f16.gguf y export/tutor_gguf/tutor_mejorado.gguf — Archivos de modelo convertidos (artefactos listos).
