"""Clasificación con el modelo fine-tuned servido por vLLM en RunPod."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI, AuthenticationError
from PIL import Image

import config
from images import pil_to_data_url

log = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        base = f"{config.CLASSIFY_URL}/v1"
        masked = (
            f"{config.RUNPOD_API_KEY[:4]}…{config.RUNPOD_API_KEY[-4:]}"
            if len(config.RUNPOD_API_KEY) >= 8 else "<short>"
        )
        log.info(
            "classify vLLM client init base_url=%s model=%s key=%s",
            base, config.CLASSIFY_MODEL, masked,
        )
        _client = AsyncOpenAI(base_url=base, api_key=config.RUNPOD_API_KEY)
    return _client


async def classify_property(pil_image: Image.Image) -> dict[str, Any]:
    """Devuelve {label, raw, error}. label ∈ CLASS_NAMES ∪ {'unknown'}.

    Usa `guided_choice` de vLLM para restringir la salida del decoder a una
    de las etiquetas permitidas. Esto garantiza el contrato del endpoint
    incluso si el modelo intenta divagar.
    """
    image_url = pil_to_data_url(pil_image)
    allowed = [*config.CLASS_NAMES, "unknown"]
    messages = [
        {"role": "system", "content": config.SYSTEM_MSG},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": config.USER_PROMPT.format(classes=", ".join(config.CLASS_NAMES))},
            ],
        },
    ]
    try:
        resp = await _get_client().chat.completions.create(
            model=config.CLASSIFY_MODEL,
            messages=messages,
            max_tokens=8,
            temperature=0.0,
            extra_body={"guided_choice": allowed},
        )
    except AuthenticationError as e:
        log.exception("Classification failed (401) on %s", config.CLASSIFY_URL)
        return {"label": "unknown", "raw": None, "error": str(e)}
    except Exception as e:
        log.exception("Classification failed")
        return {"label": "unknown", "raw": None, "error": str(e)}

    raw = (resp.choices[0].message.content or "").strip()
    log.info(
        "vLLM classify response: raw=%r finish_reason=%s usage=%s",
        raw,
        resp.choices[0].finish_reason,
        resp.usage.model_dump() if resp.usage else None,
    )
    # Normaliza: minúsculas, '_' y '-' como espacio, colapsa espacios.
    norm = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    label = "unknown"
    for name in config.CLASS_NAMES:
        canon = name.lower().replace("_", " ")
        if canon in norm:
            label = name
            break
    return {"label": label, "raw": raw, "error": None}
