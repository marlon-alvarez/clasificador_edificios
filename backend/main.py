from __future__ import annotations

import io
from pathlib import Path

import gdown
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model3.keras"

GDRIVE_FILE_ID = "1CCsaWh9hot0DGTnYickmVeb62pdQBgmG"

IMG_SIZE = 224

CLASSES = [
    "apartment", "church", "garage", "house",
    "industrial", "officebuilding", "retail", "roof",
]
CLASS_LABELS = [
    "Apartment", "Church", "Garage", "House",
    "Industrial", "Office Bldg", "Retail", "Roof",
]

app = FastAPI(title="Clasificador de Edificios")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: keras.Model | None = None


def download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading model from Google Drive ({GDRIVE_FILE_ID}) ...")
    gdown.download(url, str(MODEL_PATH), quiet=False)
    print("Download complete.")


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        download_model()
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        batch = preprocess(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "class": CLASSES[idx],
        "label": CLASS_LABELS[idx],
        "confidence": round(float(preds[idx]), 4),
        "probabilities": {
            label: round(float(p), 4)
            for label, p in zip(CLASS_LABELS, preds)
        },
    }
