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
        base = f"{config.RUNPOD_URL.rstrip('/')}/v1"
        # [TESTING] Quitar o silenciar antes de producción: expone RUNPOD_API_KEY completa.
        log.warning(
            "[TESTING] vLLM client init base_url=%s model=%s RUNPOD_API_KEY=%r",
            base,
            config.MODEL_NAME,
            config.RUNPOD_API_KEY,
        )
        _client = AsyncOpenAI(base_url=base, api_key=config.RUNPOD_API_KEY)
    return _client


def _log_auth_hint() -> None:
    log.warning(
        "[TESTING] vLLM 401 — clave que envía esta API (repr): %r | "
        "curl: curl -sS %r -H %r -H 'Content-Type: application/json' "
        "-d '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":3}'",
        config.RUNPOD_API_KEY,
        f"{config.RUNPOD_URL.rstrip('/')}/v1/chat/completions",
        f"Authorization: Bearer {config.RUNPOD_API_KEY}",
        config.MODEL_NAME,
    )


async def classify_property(pil_image: Image.Image) -> dict[str, Any]:
    """Devuelve {label, raw, error}. label ∈ CLASS_NAMES ∪ {'unknown'}."""
    image_url = pil_to_data_url(pil_image)
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
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=32,
            temperature=0.0,
        )
    except AuthenticationError as e:
        _log_auth_hint()
        log.exception("Classification failed (401)")
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
    lower = raw.lower()
    label = "unknown"
    for name in config.CLASS_NAMES:
        if name.lower() in lower:
            label = name
            break
    return {"label": label, "raw": raw, "error": None}
