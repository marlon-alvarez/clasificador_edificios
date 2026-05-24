"""Detección y blur de caras humanas con YuNet (OpenCV DNN).

YuNet es un detector basado en red neuronal incluido en OpenCV 4.5.4+
(`cv2.FaceDetectorYN`). Comparado con Haar:
  - Recall mucho mejor en caras chicas, perfiles, con gafas, contraluz.
  - Latencia comparable en CPU (~30-80 ms por imagen 1500x800).
  - Modelo ONNX de ~340 KB, descargado una vez a `api/models/` y cacheado.

Se mantiene la firma `anonymize_faces(pil) -> (pil_anonimizada, n_caras)`
para no tocar el resto del pipeline.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Modelo YuNet (face_detection_yunet_2023mar) — OpenCV Zoo.
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "face_detection_yunet_2023mar.onnx"

# Umbral conservador para capturar caras chicas/perfiles. Subir si aparecen
# falsos positivos (ventanas, lámparas) en producción.
_SCORE_THRESHOLD = 0.55
_NMS_THRESHOLD = 0.3
_TOP_K = 5000

_detector: cv2.FaceDetectorYN | None = None
_last_size: tuple[int, int] | None = None


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Descargando modelo YuNet a %s ...", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    log.info("Modelo YuNet listo (%d bytes).", _MODEL_PATH.stat().st_size)
    return _MODEL_PATH


def _get_detector(size: tuple[int, int]) -> cv2.FaceDetectorYN:
    """Devuelve el detector YuNet, reusándolo entre llamadas.

    `size` es (w, h). YuNet necesita que `setInputSize` coincida con la imagen.
    """
    global _detector, _last_size
    if _detector is None:
        model_path = _ensure_model()
        _detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            size,
            _SCORE_THRESHOLD,
            _NMS_THRESHOLD,
            _TOP_K,
        )
        _last_size = size
        return _detector
    if size != _last_size:
        _detector.setInputSize(size)
        _last_size = size
    return _detector


def _detect(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    detector = _get_detector((w, h))
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    rects: list[tuple[int, int, int, int]] = []
    for row in faces:
        # row = [x, y, w, h, lm0x, lm0y, ..., score]
        x, y, fw, fh = row[:4]
        rects.append((int(x), int(y), int(fw), int(fh)))
    return rects


def anonymize_faces(pil_image: Image.Image) -> tuple[Image.Image, int]:
    """Devuelve (imagen_anonimizada, n_caras_blureadas)."""
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rects = _detect(bgr)
    faces = 0
    for x, y, fw, fh in rects:
        pad_x = int(fw * 0.15)
        pad_y = int(fh * 0.15)
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
