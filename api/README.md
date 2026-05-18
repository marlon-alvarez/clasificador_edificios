# API — Procesamiento de imágenes de inmuebles

Servicio FastAPI **independiente** con un único endpoint `POST /processing`.

Pipeline aplicado a **todas** las imágenes:

1. Anonimiza caras humanas (mediapipe + blur Gaussiano).
2. OCR (easyocr) + extracción de direcciones, teléfonos, emails, URLs,
   precios, áreas y menciones de color.
3. Descripción de inmuebles aledaños y referencias visuales (Gemma-3-VL).
4. Color dominante (KMeans).

A la **primera** imagen (la principal del inmueble) se le añade además la
**clasificación** del tipo de inmueble usando el modelo Gemma-3-VL
fine-tuned servido por vLLM en RunPod.

Devuelve un JSON con la principal al inicio y un `summary` consolidado.

## Estructura

```
api/
  main.py          # FastAPI app + endpoint /processing
  config.py        # .env + variables de entorno y prompts del fine-tune
  .env.example     # plantilla (copiar a .env)
  .gitignore        # ignora .env y .venv
  anonymizer.py    # mediapipe + cv2 blur de caras
  classifier.py    # cliente OpenAI → vLLM (clasificación + aledaños)
  ocr.py           # easyocr + regex de campos
  colors.py        # KMeans dominant colors + paleta nombrada
  images.py        # PIL <-> base64
  requirements.txt
  .env.example
  README.md
```

## Pre-requisitos

- Python 3.10+
- Pod de vLLM con el modelo Gemma-3-VL fine-tuned ya corriendo en RunPod
  (ver `../PLAN_DEPLOY.md`). Necesitas su URL pública y la API key.

## Configuración (`.env`)

Copia la plantilla y edítala **una vez** en `api/.env`:

```bash
cd api
cp .env.example .env
# edita .env con tu RUNPOD_URL y RUNPOD_API_KEY
```

`config.py` carga automáticamente `api/.env` al arrancar. Si alguna variable
ya existe en el entorno del proceso, **gana** el valor del entorno (útil en
CI o Docker sin tocar `.env`).

| Var | Default | Descripción |
|---|---|---|
| `RUNPOD_URL` | `http://localhost:8000` | URL pública del pod vLLM (sin `/v1`) |
| `RUNPOD_API_KEY` | `EMPTY` | API key del start command de vLLM |
| `MODEL_NAME` | `gemma-3-4b-ft` | `served-model-name` configurado en vLLM |
| `CLASS_NAMES` | `apartamento,casa,local_comercial` | Clases del fine-tune (CSV) |
| `OCR_LANGS` | `es,en` | Idiomas para easyocr (CSV) |
| `RETURN_IMAGES` | `1` | `0` para no devolver base64 (respuestas más livianas) |
| `PORT` | `8080` | Puerto local de la API |

## Correr (siempre desde dentro de `api/`)

```bash
cd api

# 1. Entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate

# 2. Dependencias
pip install -r requirements.txt

# 3. Config: cp .env.example .env y edita .env (RUNPOD_URL, RUNPOD_API_KEY, …)

# 4. Levantar
python main.py
# o equivalentemente:
# uvicorn main:app --host 0.0.0.0 --port 8080
```

> **Importante:** `RUNPOD_URL` va **sin** sufijo `/v1` y **sin** trailing slash;
> `config.py` añade `/v1` internamente.
> `RUNPOD_API_KEY` es solo el secreto, **no** lleva `Bearer ` adelante.

Primer request: la primera vez que llegue tráfico, easyocr y mediapipe se
cargan en memoria (~30-60 s). Llamadas siguientes ya son rápidas.

## Probar

### Health
```bash
curl http://localhost:8080/health | jq
```

### Procesar imágenes (la primera = principal)
```bash
curl -F "images=@principal.jpg" \
     -F "images=@letrero.jpg" \
     -F "images=@fachada.jpg" \
     http://localhost:8080/processing | jq
```

### Respuesta sin base64 (más liviana)

En `.env` pon `RETURN_IMAGES=0` y reinicia la API.

## Estructura de la respuesta

```jsonc
{
  "main_image": {
    "index": 0,
    "filename": "principal.jpg",
    "role": "principal",
    "classification": { "label": "casa", "raw": "casa", "error": null },
    "dominant_colors": [{ "name": "beige", "hex": "...", "ratio": 0.42 }, ...],
    "faces_blurred": 0,
    "ocr_text": "...",
    "ocr_blocks": [...],
    "extracted_fields": { "addresses": [...], "phones": [...], ... },
    "surroundings": { "description": "...", "error": null },
    "anonymized_image_b64": "..."
  },
  "property_type": "casa",
  "additional_images": [ /* misma forma que main_image, sin classification */ ],
  "summary": {
    "property_type": "casa",
    "main_dominant_colors": ["beige", "marron", ...],
    "all_dominant_colors": [...],
    "candidate_addresses": [...],
    "candidate_phones": [...],
    "candidate_emails": [...],
    "candidate_urls": [...],
    "candidate_prices": [...],
    "candidate_areas": [...],
    "color_mentions_in_ocr": [...],
    "ocr_concatenated": "...",
    "surroundings_concatenated": "...",
    "total_images": 3,
    "total_faces_blurred": 1
  }
}
```

## Troubleshooting

- **`401 Unauthorized` al clasificar** → revisa que `RUNPOD_API_KEY` coincida
  con el `--api-key` que pusiste en el start command de vLLM.
- **`Connection refused`** → el pod vLLM está apagado o `RUNPOD_URL` está mal.
  Verifica (sustituye URL y clave):  
  `curl -sS "https://TU-POD-8000.proxy.runpod.net/v1/models" -H "Authorization: Bearer TU_CLAVE"`.
- **Mediapipe falla en Mac M1/M2** → usa Python 3.10 u 11 (mediapipe aún no
  publica wheels para 3.13+).
- **Easyocr descarga modelos en cada arranque** → cachea en `~/.EasyOCR/`,
  el primer arranque baja ~60 MB.
- **Latencia alta** → cada imagen secundaria hace una llamada extra al VLM
  para describir aledaños. Si necesitas velocidad, podemos exponer un flag
  para desactivarlo.
