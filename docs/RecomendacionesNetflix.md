# Recomendaciones de Netflix y Amazon

## Como generan recomendaciones

- **Amazon:** grandes sistemas de "candidate generation" basados en co-compra/vistas y reglas de negocio; luego un potente ranker personalizado.
- **Netflix:** combina factorization, modelos de secuencia y ensembles; pipeline dividido en generación de candidatos y ranking final.

## Methodos comunmente usados

- Collaborative filtering (item/item, user/user)
- Matrix factorization (ALS, SVD)
- Modelos basados en contenido (atributos del ítem)
- Modelos secuenciales (RNN/Transformer) para sesión
- Sistemas híbridos y aprendizaje por refuerzo (bandits)

## Como lo podriamos hacer

1. Recolectar logs: user_id, item_id, evento, timestamp.
2. Baseline: recomendar top-popular y top-por-categoría.
3. Implementar item-item por similitud (coseno) como segundo baseline.
4. Entrenar ALS (implicit) para obtener embeddings y generar candidatos.
5. Construir ranker (LightGBM) con features: similaridad, popularidad, recencia, contexto.
6. Evaluar offline (Precision@K, NDCG) y, si es posible, A/B online.

## Tecnologias que se pueden usar

- Python, pandas, scikit-learn, implicit, LightGBM, FAISS (ANN).
- Kafka/DB simple para eventos; Redis para cache.
