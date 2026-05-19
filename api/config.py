"""Configuración leída desde variables de entorno y archivo `.env`.

Dos endpoints vLLM en el mismo pod GPU:
  - CLASSIFY_*   → fine-tune de clasificación (gemma-3-4b-ft).
  - DESCRIBE_*   → modelo base para descripción/OCR/horario (gemma-3-4b-it).
La API key es la misma para ambos (se levantan con el mismo --api-key).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

# Carga `api/.env` al importar este módulo.
# `override=True` porque RunPod inyecta su propia env `RUNPOD_API_KEY`
# (la API key de su plataforma) que tapaba la nuestra. El `.env` manda.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _url(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().rstrip("/")


# Clasificación (fine-tune, guided_choice). Pod GPU :8000.
CLASSIFY_URL = _url("CLASSIFY_URL", "http://localhost:8000")
CLASSIFY_MODEL = os.environ.get("CLASSIFY_MODEL", "gemma-3-4b-ft").strip()

# Descripción / OCR / parse de horario. Pod GPU :8001 (modelo base).
DESCRIBE_URL = _url("DESCRIBE_URL", "http://localhost:8001")
DESCRIBE_MODEL = os.environ.get("DESCRIBE_MODEL", "gemma-3-4b-it").strip()

# Misma API key para los dos procesos vLLM.
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "EMPTY").strip()

CLASS_NAMES = [
    c.strip() for c in os.environ.get(
        "CLASS_NAMES",
        "apartamento,casa,local_comercial",
    ).split(",") if c.strip()
]

# Prompts del clasificador. Una sola tarea: devolver UNA palabra exacta.
# Salidas válidas: las clases de CLASS_NAMES o `unknown` si no hay certeza.
SYSTEM_MSG = (
    "Clasificas fotos de inmuebles. Respondes con UNA sola palabra exacta, "
    "sin explicación ni puntuación."
)
USER_PROMPT = (
    "¿Qué tipo de inmueble se ve en la foto? "
    "Responde con UNA sola palabra, exactamente una de: {classes}. "
    "Si la imagen no permite decidir con certeza, responde: unknown."
)
