"""
FastAPI app — POST /processing.

Pipeline:
  1. Recibe un array de imágenes (multipart/form-data, campo `images`).
  2. Anonimiza caras humanas en TODAS.
  3. A TODAS se les aplica: OCR + extracción de campos (direcciones,
     teléfonos, etc.) + descripción de inmuebles aledaños / referencias
     visuales + colores dominantes.
  4. La primera = principal del inmueble → además se clasifica con
     Gemma-3-VL fine-tuned.
  5. Devuelve JSON estructurado: principal con clasificación primero, luego
     secundarias y un summary consolidado de todo el contexto.

Configuración: `api/.env` (cargado en `config.py`) o variables de entorno.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

import config
from anonymizer import anonymize_faces
from classifier import classify_property, describe_surroundings
from colors import dominant_colors
from images import load_pil, pil_to_base64
from ocr import extract_fields, ocr_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Real Estate Image Processing API",
    description=(
        "Anonimiza caras, clasifica el inmueble principal con Gemma-3-VL y "
        "extrae información de las imágenes secundarias vía OCR."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "runpod_url": config.RUNPOD_URL,
        "classes": config.CLASS_NAMES,
        "ocr_langs": config.OCR_LANGS,
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

    # 2. Anonimización global
    anonymized = [(name, *anonymize_faces(img)) for name, img in loaded]

    # 3. Procesar TODAS las imágenes (OCR + campos + aledaños + colores)
    processed: list[dict[str, Any]] = []
    agg: dict[str, set[str]] = {
        "addresses": set(), "colors_ocr": set(),
    }
    all_text: list[str] = []
    all_surroundings: list[str] = []
    all_dominant_colors: set[str] = set()

    for i, (name, img, n_faces) in enumerate(anonymized):
        full_text, blocks, detailed = ocr_image(img)
        fields = extract_fields(full_text, blocks)
        surroundings = describe_surroundings(img)
        colors = dominant_colors(img, k=4)

        agg["addresses"].update(fields["addresses"])
        agg["colors_ocr"].update(fields["color_mentions"])
        all_dominant_colors.update(colors)
        if full_text:
            all_text.append(full_text)
        if surroundings.get("description"):
            all_surroundings.append(surroundings["description"])

        item: dict[str, Any] = {
            "index": i,
            "filename": name,
            "role": "principal" if i == 0 else "secundaria",
            "faces_blurred": n_faces,
            "dominant_colors": colors,
            "ocr_text": full_text,
            "ocr_blocks": detailed,
            "extracted_fields": fields,
            "surroundings": surroundings,
        }
        if config.RETURN_IMAGES:
            item["anonymized_image_b64"] = pil_to_base64(img)
        processed.append(item)

    # 4. Clasificar la principal (índice 0) y enriquecer su bloque
    main_block = processed[0]
    classification = classify_property(anonymized[0][1])
    main_block["classification"] = classification

    # 5. Respuesta — primero la principal con su clasificación.
    return {
        "main_image": main_block,
        "property_type": classification["label"],
        "additional_images": processed[1:],
        "summary": {
            "property_type": classification["label"],
            "main_dominant_colors": main_block["dominant_colors"],
            "all_dominant_colors": sorted(all_dominant_colors),
            "candidate_addresses": sorted(agg["addresses"]),
            "color_mentions_in_ocr": sorted(agg["colors_ocr"]),
            "ocr_concatenated": " | ".join(all_text),
            "surroundings_concatenated": " | ".join(all_surroundings),
            "total_images": len(images),
            "total_faces_blurred": sum(n for _, _, n in anonymized),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), reload=False)
