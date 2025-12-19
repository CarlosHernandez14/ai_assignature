# Proyecto RAG local

Este proyecto usa RAG (Retrieval-Augmented Generation) para responder preguntas con ayuda de documentos reales (artículos web, PDFs, etc.). En palabras simples: primero se guardan textos en una “biblioteca” vectorial y, cuando haces una pregunta, el sistema busca los pasajes más relevantes y se los pasa al modelo para que responda mejor y con contexto.

La idea es práctica: juntar fuentes confiables, organizarlas, y obtener respuestas más precisas que un modelo “a ciegas”.

## ¿Qué es RAG en pocas palabras?

- RAG combina dos cosas:
  1) Recuperación: busca trozos de texto relevantes en una base vectorial.
  2) Generación: el modelo de lenguaje escribe la respuesta usando esos chunks como apoyo.

## Flujo del proyecto

1) Reunir textos
   - Se pueden “raspar” artículos de la web (RSS/URLs) o leer PDFs locales.
2) Preparar y separar
   - Los textos se limpian y se cortan en fragmentos manejables (chunks).
3) Crear la base vectorial
   - Se convierten los fragmentos en vectores con un modelo de embeddings y se guardan en Chroma.
4) Preguntar y generar informe
   - Para cada pregunta, se recuperan los fragmentos más útiles, se arma un prompt y el modelo genera respuestas ordenadas en un informe.


## Cómo se conectan las piezas

- scrape_corpus.py / ingest_pdfs.py → generan un corpus en JSONL.
- chunk_jsonl.py → corta ese corpus en trozos legibles por el recuperador.
- chroma_vector_embedings.py → crea/conecta la base vectorial (Chroma) con embeddings de Ollama.
- rag_ollama.py → hace las preguntas, recupera contexto y genera un informe final en Markdown.


## Archivos y carpetas

- INFORME_RAG_MISTRAL.md  
  Ejemplo de informe ya generado por el sistema con preguntas y respuestas.

- rag_ollama.py  
  Orquesta el RAG: lee preguntas, recupera pasajes relevantes y produce un informe final. Usa un modelo de chat local (Ollama).

- chroma_vector_embedings.py  
  Crea o conecta una base vectorial (Chroma). Genera embeddings con Ollama (por defecto “mxbai-embed-large”) y arma el recuperador (k pasajes).

- chunk_jsonl.py  
  Limpia y divide textos largos en fragmentos (chunks) con solapamiento. Útil para mejorar la recuperación.

- scrape_corpus.py  
  Descarga artículos desde feeds RSS/Atom o URLs sueltas. Extrae título/autor/texto con trafilatura y guarda todo como JSONL.

- ingest_pdfs.py  
  Lee PDFs locales (por documento o por página), extrae texto y metadatos, y los guarda como JSONL.

- preguntas.txt  
  Lista de preguntas de investigación (una por línea). El generador de informes las usa como entrada.

- feeds.txt / urls.txt  
  Listas editables de fuentes web (RSS y enlaces directos) alineadas con los temas del proyecto.

- corpus/  
  Carpeta para guardar los artículos ya extraídos (JSONL). Sirve como base cruda antes de trocear.

- vectorstore/  
  Carpeta donde Chroma guarda la base vectorial persistente (índice de embeddings).


## Modelos y configuración
- El proyecto usa Ollama local tanto para embeddings como para el modelo de chat.  
- Por defecto, el modelo de chat es “mistral-nemo” y el de embeddings “mxbai-embed-large” (configurable por variables de entorno si se desea).
- La recuperación busca varios pasajes (k) y prioriza diversidad (MMR) para cubrir distintas partes del tema.