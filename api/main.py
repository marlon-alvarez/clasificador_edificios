"""
FastAPI app — POST /processing.

Pipeline:
  1. Recibe un array de imágenes (multipart/form-data, campo `images`).
  2. Anonimiza caras humanas en TODAS.
  3. Llamadas al VLM en paralelo (N llamadas totales):
     - Principal (índice 0)        → 1 clasificación con Gemma-3-VL fine-tuned.
     - Cada imagen adicional       → 1 extracción combinada (OCR + alrededores
       + horario) que devuelve JSON estructurado en una sola request.
  4. Colores dominantes y campos (direcciones, menciones de color) se calculan
     local en CPU.
  5. Consolida horario: primer horario detectado entre las extras, o un default
     (L-V 08:00-17:00, Sábado 08:00-12:00, Domingo cerrado) si nadie lo trae.
  6. Devuelve JSON estructurado: principal con clasificación primero, luego
     secundarias, horario, y un summary consolidado de todo el contexto.

Configuración: `api/.env` (cargado en `config.py`) o variables de entorno.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

import config
from anonymizer import anonymize_faces
from classifier import classify_property
from colors import dominant_colors
from extractor import (
    consolidate_schedule,
    describe_surroundings,
    extract_schedule,
)
from images import load_pil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Real Estate Image Processing API",
    description=(
        "Anonimiza caras, clasifica el inmueble principal con Gemma-3-VL y "
        "extrae OCR + alrededores + horario de las imágenes adicionales en "
        "una sola llamada combinada por imagen."
    ),
    version="1.1.0",
)


@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "runpod_url": config.RUNPOD_URL,
        "classes": config.CLASS_NAMES,
    }


@app.post("/processing")
async def processing(
    images: list[UploadFile] = File(
        ..., description="Array de imágenes. La primera es la principal del inmueble."
    ),
) -> dict[str, Any]:
    if not images:
        raise HTTPException(400, "Se requiere al menos una imagen.")

    # 1. Cargar
    loaded: list[tuple[str, Any]] = []
    for f in images:
        data = await f.read()
        if not data:
            raise HTTPException(400, f"Archivo vacío: {f.filename}")
        try:
            loaded.append((f.filename or f"image_{len(loaded)}", load_pil(data)))
        except Exception as e:
            raise HTTPException(400, f"No se pudo decodificar '{f.filename}': {e}")

    # 2. Anonimización global (CPU local)
    anonymized = [(name, *anonymize_faces(img)) for name, img in loaded]

    # 3. Llamadas al VLM en paralelo, una por tipo de extracción:
    #    - 1 clasificación (sólo imagen principal)
    #    - (N-1) extracciones de horario (una por extra)
    #    - (N-1) descripciones de entorno (una por extra)
    extras = anonymized[1:]
    classify_coro = classify_property(anonymized[0][1])
    schedule_coros = [extract_schedule(img) for _, img, _ in extras]
    surroundings_coros = [describe_surroundings(img) for _, img, _ in extras]

    schedules_per_extra, surroundings_per_extra, classification = await asyncio.gather(
        asyncio.gather(*schedule_coros),
        asyncio.gather(*surroundings_coros),
        classify_coro,
    )

    # 4. Ensamblado por imagen (colores locales + datos extraídos)
    processed: list[dict[str, Any]] = []
    all_surroundings: list[str] = []
    all_dominant_colors: set[str] = set()

    for i, (name, img, n_faces) in enumerate(anonymized):
        colors = dominant_colors(img, k=4)
        all_dominant_colors.update(colors)

        if i == 0:
            surroundings_text = ""
        else:
            surroundings_text = surroundings_per_extra[i - 1]
            if surroundings_text:
                all_surroundings.append(surroundings_text)

        item: dict[str, Any] = {
            "index": i,
            "filename": name,
            "role": "principal" if i == 0 else "secundaria",
            "faces_blurred": n_faces,
            "dominant_colors": colors,
            "surroundings": {"description": surroundings_text},
        }
        processed.append(item)

    # 5. Consolidar horario (primer detectado entre las extras, o default)
    schedule = consolidate_schedule(schedules_per_extra, base_index=1)

    # 6. Clasificación a la principal
    main_block = processed[0]
    main_block["classification"] = classification

    # 7. Respuesta
    return {
        "main_image": main_block,
        "property_type": classification["label"],
        "schedule": schedule,
        "additional_images": processed[1:],
        "summary": {
            "property_type": classification["label"],
            "main_dominant_colors": main_block["dominant_colors"],
            "all_dominant_colors": sorted(all_dominant_colors),
            "surroundings_concatenated": " | ".join(all_surroundings),
            "total_images": len(images),
            "total_faces_blurred": sum(n for _, _, n in anonymized),
            "schedule_source": schedule["source"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), reload=False)
