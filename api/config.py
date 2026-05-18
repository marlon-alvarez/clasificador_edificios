"""Configuración leída desde variables de entorno y archivo `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

# Carga `api/.env` al importar este módulo (antes de leer os.environ).
# Las variables ya definidas en el entorno del proceso tienen prioridad.
load_dotenv(Path(__file__).resolve().parent / ".env")

RUNPOD_URL = os.environ.get("RUNPOD_URL", "http://localhost:8000").strip().rstrip("/")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "EMPTY").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-3-4b-ft").strip()

CLASS_NAMES = [
    c.strip() for c in os.environ.get(
        "CLASS_NAMES",
        "apartamento,casa,local_comercial",
    ).split(",") if c.strip()
]

# Prompts idénticos a los del fine-tune (gemma3_vl_clasificador_inmuebles.py)
SYSTEM_MSG = (
    "You are an expert image classifier for real estate properties. "
    "You classify images into exactly one category. "
    "Only respond with the category name, nothing else."
)
USER_PROMPT = (
    "Classify this image into exactly one of the following categories: {classes}\n"
    "Respond with ONLY the category name."
)
