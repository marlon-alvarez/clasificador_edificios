"""Helpers de codificación / decodificación de imágenes."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageOps

VLM_MAX_SIDE = 1536
VLM_JPEG_QUALITY = 90


def load_pil(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def downscale_for_vlm(pil_image: Image.Image, max_side: int = VLM_MAX_SIDE) -> Image.Image:
    w, h = pil_image.size
    longest = max(w, h)
    if longest <= max_side:
        return pil_image
    ratio = max_side / longest
    return pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def pil_to_base64(pil_image: Image.Image, fmt: str = "JPEG", quality: int = VLM_JPEG_QUALITY) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pil_to_data_url(pil_image: Image.Image) -> str:
    return f"data:image/jpeg;base64,{pil_to_base64(pil_image)}"
