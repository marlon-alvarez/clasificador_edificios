"""
FastAPI app — POST /processing.

Pipeline (una llamada al modelo por acción):
  1. Recibe un array de imágenes (multipart/form-data, campo `images`).
  2. Anonimiza caras humanas en TODAS (CPU local, OpenCV Haar).
  3. Llamadas al VLM en paralelo:
     - Principal (índice 0)  → clasificación casa/apartamento/local_comercial.
     - Cada imagen           → descripción OCR (colores, letreros, horarios
                               escritos, pistas del entorno) para domiciliarios.
  4. Llamada de texto al LLM con la descripción concatenada → horario JSON
     o `null`. Si `null` → `DEFAULT_SCHEDULE`.
  5. Devuelve JSON estructurado.

Configuración: `api/.env` (cargado en `config.py`) o variables de entorno.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from anonymizer import anonymize_faces
from classifier import classify_property
from extractor import (
    consolidate_schedule,
    extract_description,
    parse_schedule,
)
from images import downscale_for_vlm, load_pil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Real Estate Image Processing API",
    description=(
        "Anonimiza caras, clasifica el inmueble principal con Gemma-3-VL, "
        "extrae una descripción OCR por imagen (colores, letreros, horarios, "
        "entorno) y consolida el horario de atención en JSON."
    ),
    version="2.0.0",
)

# CORS abierto a todos los orígenes. Endurecer en producción si la API se
# expone fuera de un cliente controlado.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # con "*" no se permiten credentials (spec CORS)
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "classify": {"url": config.CLASSIFY_URL, "model": config.CLASSIFY_MODEL},
        "describe": {"url": config.DESCRIBE_URL, "model": config.DESCRIBE_MODEL},
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

    # 2. Anonimización local (Haar a resolución original) y downscale para el VLM.
    #    Orden obligatorio por privacidad: las caras se difuminan antes de cualquier
    #    salida hacia el pod GPU, y el downscale ocurre sobre la imagen ya anonimizada.
    anonymized: list[tuple[str, Any, int]] = []
    for name, img in loaded:
        anon, n_faces = anonymize_faces(img)
        anonymized.append((name, downscale_for_vlm(anon), n_faces))

    # 3. Visión en paralelo:
    #    - Clasificación de la principal (1 llamada).
    #    - Descripción por imagen, con prompt según rol principal/secundaria.
    classify_coro = classify_property(anonymized[0][1])
    description_coros = [
        extract_description(img, role="main" if i == 0 else "secondary")
        for i, (_, img, _) in enumerate(anonymized)
    ]

    classification, descriptions = await asyncio.gather(
        classify_coro,
        asyncio.gather(*description_coros),
    )

    # 4. Concatenar descripciones y pedir horario al LLM (1 llamada, solo texto).
    concatenated = " | ".join(d for d in descriptions if d)
    parsed_schedule = await parse_schedule(concatenated)
    schedule = consolidate_schedule(parsed_schedule)

    # 5. Armar respuesta por imagen.
    per_image: list[dict[str, Any]] = []
    for i, (name, _, n_faces) in enumerate(anonymized):
        per_image.append({
            "index": i,
            "filename": name,
            "role": "principal" if i == 0 else "secundaria",
            "faces_blurred": n_faces,
            "description": descriptions[i],
        })

    return {
        "property_type": classification["label"],
        "classification": classification,
        "images": per_image,
        "schedule": schedule,
        "summary": {
            "total_images": len(images),
            "total_faces_blurred": sum(n for _, _, n in anonymized),
            "schedule_source": schedule["source"],
            "description_concatenated": concatenated,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), reload=False)
