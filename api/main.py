"""
FastAPI app — POST /processing.

Pipeline:
  1. Recibe un array de imágenes (multipart/form-data, campo `images`).
  2. Anonimiza caras humanas en TODAS (CPU local, OpenCV Haar).
  3. Llamadas al VLM en paralelo:
     - Principal (índice 0)  → clasificación + descripción del inmueble.
     - Cada extra            → horario + descripción del entorno.
  4. Consolida horario: primer horario detectado entre las extras, o un default
     (L-V 08:00-17:00, Sábado 08:00-12:00, Domingo cerrado) si nadie lo trae.
  5. Devuelve JSON estructurado: principal con clasificación + descripción,
     secundarias con surroundings, horario, y un summary consolidado.

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
from extractor import (
    consolidate_schedule,
    describe_site,
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

    # 3. Llamadas al VLM en paralelo:
    #    - Principal: clasificación + descripción del inmueble.
    #    - Cada extra: horario + descripción del entorno.
    extras = anonymized[1:]
    classify_coro = classify_property(anonymized[0][1])
    site_coro = describe_site(anonymized[0][1])
    schedule_coros = [extract_schedule(img) for _, img, _ in extras]
    surroundings_coros = [describe_surroundings(img) for _, img, _ in extras]

    classification, site_description, schedules_per_extra, surroundings_per_extra = await asyncio.gather(
        classify_coro,
        site_coro,
        asyncio.gather(*schedule_coros),
        asyncio.gather(*surroundings_coros),
    )

    # 4. Bloque de la imagen principal (clasificación + descripción del inmueble)
    main_name, _, main_faces = anonymized[0]
    main_image = {
        "index": 0,
        "filename": main_name,
        "role": "principal",
        "faces_blurred": main_faces,
        "classification": classification,
        "site_description": site_description,
    }

    # 5. Imágenes adicionales (solo surroundings)
    additional_images: list[dict[str, Any]] = []
    all_surroundings: list[str] = []
    for i, (name, _, n_faces) in enumerate(extras, start=1):
        surroundings_text = surroundings_per_extra[i - 1]
        if surroundings_text:
            all_surroundings.append(surroundings_text)
        additional_images.append({
            "index": i,
            "filename": name,
            "role": "secundaria",
            "faces_blurred": n_faces,
            "surroundings": surroundings_text,
        })

    # 6. Consolidar horario (primer detectado entre las extras, o default)
    schedule = consolidate_schedule(schedules_per_extra, base_index=1)

    # 7. Respuesta
    return {
        "property_type": classification["label"],
        "main_image": main_image,
        "additional_images": additional_images,
        "schedule": schedule,
        "summary": {
            "total_images": len(images),
            "total_faces_blurred": sum(n for _, _, n in anonymized),
            "schedule_source": schedule["source"],
            "surroundings_concatenated": " | ".join(all_surroundings),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), reload=False)
