"""Color dominante de la imagen (solo HEX) + paleta para detectar menciones en OCR."""

from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Paleta usada por extractor.py para detectar menciones de color en el texto.
PALETTE: dict[str, tuple[int, int, int]] = {
    "blanco": (240, 240, 240),
    "negro": (20, 20, 20),
    "gris": (128, 128, 128),
    "rojo": (200, 30, 30),
    "naranja": (230, 130, 40),
    "amarillo": (235, 210, 60),
    "verde": (60, 160, 70),
    "azul": (60, 100, 200),
    "morado": (130, 70, 180),
    "rosa": (230, 130, 170),
    "marron": (110, 70, 40),
    "beige": (220, 200, 170),
    "crema": (245, 235, 210),
    "terracota": (200, 110, 80),
}


def dominant_colors(pil_image: Image.Image, k: int = 4) -> list[str]:
    """Devuelve los k colores dominantes ordenados por frecuencia, solo HEX."""
    img = pil_image.convert("RGB").copy()
    img.thumbnail((200, 200))
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    if len(arr) < k:
        k = max(1, len(arr))
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(arr)
    counts = np.bincount(km.labels_, minlength=k)
    out: list[str] = []
    for idx in np.argsort(-counts):
        r, g, b = (int(v) for v in km.cluster_centers_[idx])
        out.append("#{:02x}{:02x}{:02x}".format(r, g, b))
    return out
