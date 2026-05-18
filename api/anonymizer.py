"""Detección y blur de caras humanas usando OpenCV (Haar cascades).

Se usan cascadas Haar incluidas en `opencv-python`/`opencv-python-headless`
(`cv2.data.haarcascades`), por lo que no hace falta descargar modelos.
Cubrimos frontales + perfiles + perfiles invertidos.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_frontal = None
_profile = None


def _get_cascades():
    global _frontal, _profile
    if _frontal is None:
        base = cv2.data.haarcascades
        _frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        _profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if _frontal.empty() or _profile.empty():
            raise RuntimeError("No se pudieron cargar las cascadas Haar de OpenCV.")
    return _frontal, _profile


def _detect(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    frontal, profile = _get_cascades()
    rects: list[tuple[int, int, int, int]] = []
    for cascade in (frontal, profile):
        for x, y, w, h in cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        ):
            rects.append((int(x), int(y), int(w), int(h)))
    # Perfil mirando hacia el otro lado: voltear y volver a mapear coords
    flipped = cv2.flip(gray, 1)
    W = gray.shape[1]
    for x, y, w, h in profile.detectMultiScale(
        flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    ):
        rects.append((int(W - x - w), int(y), int(w), int(h)))
    return rects


def anonymize_faces(pil_image: Image.Image) -> tuple[Image.Image, int]:
    """Devuelve (imagen_anonimizada, n_caras_blureadas)."""
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rects = _detect(gray)
    faces = 0
    for x, y, fw, fh in rects:
        pad_x = int(fw * 0.10)
        pad_y = int(fh * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        if x2 <= x1 or y2 <= y1:
            continue
        region = rgb[y1:y2, x1:x2]
        k = max(31, (min(region.shape[:2]) // 4) | 1)
        rgb[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
        faces += 1
    return Image.fromarray(rgb), faces
