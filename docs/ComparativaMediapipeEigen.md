# Comparativa Mediapipe vs Eigenface/LBPH/FisherFace

> Comparativa entre EigenFace, FisherFace, LBPH y MediaPipe Face Mesh para detectar emociones, con lista práctica de puntos y medidas.

---

## ¿Qué método usaría y por qué?

- **Elección:** MediaPipe Face Mesh.
- **Por qué:** ofrece landmarks 3D en tiempo real que permiten medir movimientos faciales (cejas, ojos, boca) y su dinámica, lo que es clave para detectar emociones.

---

## Diferencias clave con los métodos clásicos

- **EigenFace / FisherFace:** trabajan la apariencia global; sensibles a iluminación y poca capacidad para capturar movimientos faciales finos.
- **LBPH:** captura textura local (buena contra iluminación), pero no modela relaciones geométricas entre puntos faciales.
- **MediaPipe:** modelo geométrico y dinámico; mejor para emociones.

---

## Puntos mínimos a extraer (lista rápida)

1. Ceja izquierda / derecha (altura y pendiente).
2. Párpados superior e inferior (ambos ojos).
3. Comisuras de la boca (izq/der) y centro de labios.
4. Punta/base de la nariz y aletas.
5. Mentón/jaw y contorno lateral (para normalizar).

---

## Medidas propuestas (fáciles de calcular)

- Normalizar por **IOD (distancia entre ojos)** o **alto de cara**.
- `Apertura_ojo = (y_sup - y_inf) / IOD`
- `Elevacion_ceja = (y_ceja - y_ojo) / face_height`
- `Apertura_boca = (y_inf_lab - y_sup_lab) / face_height`
- `Anchura_sonrisa = (x_derecha - x_izquierda) / IOD`
- `Asimetria_boca = |y_corner_r - y_corner_l| / face_height`
- Usar **derivadas temporales** (cambio/frame) para detectar reacciones rápidas.

---

## Pipeline mínimo en 3 líneas

1. Extraer landmarks (MediaPipe). 2. Calcular medidas normalizadas y sus cambios en ventana corta. 3. Clasificar (modelo simple) + suavizar salida.

---

## Consejo práctico final

- Para una tarea académica corta, extrae 6–10 medidas geométricas normalizadas y prueba un clasificador SVM/MLP; añade LSTM si quieres modelar secuencia.
