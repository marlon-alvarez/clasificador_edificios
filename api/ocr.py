"""OCR con easyocr + extracción de campos relevantes para inmuebles."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from PIL import Image

import config
from colors import PALETTE

log = logging.getLogger(__name__)

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # import diferido: easyocr es pesado

        log.info("Loading easyocr reader (langs=%s)...", config.OCR_LANGS)
        _reader = easyocr.Reader(config.OCR_LANGS, gpu=False)
    return _reader


# ── Regex de campos típicos ──
ADDRESS_RE = re.compile(
    r"\b("
    r"(?:calle|cl|cll|carrera|cra|kr|krra|avenida|av|avda|"
    r"diagonal|dg|diag|transversal|tv|trans|autopista|circular|circunvalar)\.?"
    r"[\s\.\-#°ºoNn]*"
    r"\d{1,4}[a-z]?"
    r"(?:\s*(?:bis|sur|norte|este|oeste))?"
    r"(?:\s*(?:#|n[°ºo]\.?|no\.?)\s*\d{1,4}[a-z]?)?"
    r"(?:\s*-\s*\d{1,4})?"
    r")",
    re.IGNORECASE,
)


def extract_fields(text: str, blocks: list[str]) -> dict[str, Any]:
    text_norm = " ".join(text.split())
    addresses = sorted({m.group(1).strip() for m in ADDRESS_RE.finditer(text_norm)})

    lower = text_norm.lower()
    color_mentions = sorted({c for c in PALETTE if re.search(rf"\b{c}\b", lower)})

    return {
        "addresses": addresses,
        "color_mentions": color_mentions,
    }


def ocr_image(pil_image: Image.Image) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Devuelve (texto_completo, bloques_planos, bloques_detallados)."""
    arr = np.array(pil_image.convert("RGB"))
    raw = _get_reader().readtext(arr)
    blocks: list[str] = []
    detailed: list[dict[str, Any]] = []
    for bbox, txt, conf in raw:
        if not txt or not txt.strip():
            continue
        clean = txt.strip()
        blocks.append(clean)
        detailed.append({
            "text": clean,
            "confidence": round(float(conf), 3),
            "bbox": [[int(x), int(y)] for x, y in bbox],
        })
    return " ".join(blocks), blocks, detailed
