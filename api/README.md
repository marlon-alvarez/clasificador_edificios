# API — Clasificación de inmuebles y horarios de operación

Servicio FastAPI que recibe fotos tomadas por domiciliarios y devuelve:

1. **Tipo de inmueble** (casa, apartamento, local_comercial) clasificado con
   un modelo Gemma-3-VL fine-tuned servido por vLLM.
2. **Horario de operación** estimado a partir de pistas visuales (letreros,
   horarios escritos, entorno) extraídas por un VLM y consolidadas por un LLM.
3. **Caras humanas anonimizadas** (blur local con OpenCV **YuNet**) antes de
   salir hacia cualquier servicio externo.

Es la pieza de inferencia del proyecto **Sistema de Clasificación Automática
de Tipo de Inmueble y Asignación de Horarios de Operación mediante Computer
Vision** (ver contexto al final).

## Pipeline

Para cada request a `POST /processing`:

1. Carga las imágenes (multipart, campo `images`; la primera es la principal).
2. Anonimiza caras humanas en **todas** las imágenes (OpenCV **YuNet**, local CPU).
3. En paralelo contra el pod GPU:
   - Clasifica la imagen principal → `casa | apartamento | local_comercial | unknown`.
   - Describe cada imagen (colores, letreros, horarios, entorno) con el VLM base.
4. Concatena descripciones y pide al LLM un horario JSON. Si no hay evidencia
   suficiente, cae a `DEFAULT_SCHEDULE`.
5. Devuelve la respuesta consolidada.

## Estructura

```
api/
  main.py          # FastAPI app + endpoint /processing
  config.py        # carga .env, URLs y prompts del fine-tune
  anonymizer.py    # OpenCV YuNet (DNN) blur de caras
  models/          # creado en runtime: cachea face_detection_yunet_*.onnx (~340 KB)
  classifier.py    # cliente OpenAI → vLLM clasificador (guided_choice)
  extractor.py     # descripción VLM + parsing/consolidación de horario
  images.py        # PIL <-> base64, downscale para el VLM
  requirements.txt
  .env.example
```

## Pre-requisitos

- Python 3.10 u 11 (mediapipe/OpenCV no publican wheels estables para 3.13+).
- Pod RunPod con **dos** procesos vLLM en el mismo GPU:
  - `:8000` → `gemma-3-4b-ft` (fine-tune clasificador).
  - `:8001` → `gemma-3-4b-it` (modelo base para descripción/OCR/horario).
- Ver `PLAN_DEPLOY_RUNPOD.md` para el start command y el setup del pod.

## Configuración (`.env`)

```bash
cd api
cp .env.example .env
# edita .env con las URLs de tu pod y la API key
```

| Var | Default | Descripción |
|---|---|---|
| `CLASSIFY_URL` | `http://localhost:8000` | URL pública del pod vLLM clasificador (sin `/v1`) |
| `CLASSIFY_MODEL` | `gemma-3-4b-ft` | `served-model-name` del fine-tune |
| `DESCRIBE_URL` | `http://localhost:8001` | URL pública del pod vLLM base |
| `DESCRIBE_MODEL` | `gemma-3-4b-it` | `served-model-name` del modelo base |
| `RUNPOD_API_KEY` | `EMPTY` | API key compartida por ambos procesos vLLM |
| `CLASS_NAMES` | `apartamento,casa,local_comercial` | Clases del fine-tune (CSV) |
| `PORT` | `8080` | Puerto local de la API |

> Las URLs van **sin** sufijo `/v1` y **sin** trailing slash; `config.py` añade
> `/v1` internamente. `RUNPOD_API_KEY` es solo el secreto, no lleva `Bearer`.

## Correr (desde dentro de `api/`)

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
# equivalente: uvicorn main:app --host 0.0.0.0 --port 8080
```

## Endpoints

### `GET /health`

Estado y configuración cargada.

```bash
curl http://localhost:8080/health | jq
```

### `POST /processing`

Recibe una lista de imágenes. La primera es la principal.

```bash
curl -F "images=@principal.jpg" \
     -F "images=@letrero.jpg" \
     -F "images=@fachada.jpg" \
     http://localhost:8080/processing | jq
```

Respuesta (forma resumida):

```jsonc
{
  "property_type": "casa",
  "classification": { "label": "casa", "raw": "casa", "error": null },
  "images": [
    {
      "index": 0,
      "filename": "principal.jpg",
      "role": "principal",
      "faces_blurred": 1,
      "description": "..."
    }
    // ...
  ],
  "schedule": {
    "source": "vlm" | "default",
    "weekly": { "mon": "09:00-19:00", ... }
  },
  "summary": {
    "total_images": 3,
    "total_faces_blurred": 1,
    "schedule_source": "vlm",
    "description_concatenated": "..."
  }
}
```

## Troubleshooting

- **`401 Unauthorized`** → `RUNPOD_API_KEY` no coincide con el `--api-key` del
  start command de vLLM.
- **`Connection refused`** → el pod está apagado o `CLASSIFY_URL` / `DESCRIBE_URL`
  están mal. Verificá:
  `curl -sS "https://TU-POD-8000.proxy.runpod.net/v1/models" -H "Authorization: Bearer TU_CLAVE"`.
- **Primer request lento** → en el primer arranque se descarga el modelo YuNet
  (~340 KB) a `api/models/` y se inicializan los clientes (~10-30 s).
  Llamadas siguientes ya son rápidas. Si el server no tiene salida a internet,
  pre-bajar `face_detection_yunet_2023mar.onnx` desde
  [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  y copiarlo a `api/models/` durante el build/deploy.
- **Caras chicas o de perfil no se difuminan** → bajar `_SCORE_THRESHOLD` en
  `anonymizer.py` (default `0.55`) hasta `0.4`. Si en cambio aparecen falsos
  positivos (ventanas, lámparas) subirlo a `0.7-0.8`. En panorámicas 360° con
  caras muy deformadas o distantes el recall sigue siendo limitado; ahí
  conviene complementar con un detector de personas y blurear la cabeza.
- **Latencia alta** → cada imagen secundaria hace una llamada extra al VLM
  para describir el entorno; reducir el número de imágenes baja el tiempo
  proporcionalmente.

---

## Contexto del proyecto

**Sistema de Clasificación Automática de Tipo de Inmueble y Asignación de
Horarios de Operación mediante Computer Vision para Optimización de Entregas
en Logística de Última Milla.**

**Organización**

- **Nombre:** Inter Rapidísimo S.A.
- **Sector:** Logística y mensajería.
- **Descripción:** Empresa colombiana líder en mensajería especializada en
  entregas de última milla (~10,000 entregas diarias), en proceso de
  transformación digital mediante **Inter App**, aplicación móvil en Flutter
  para domiciliarios no expertos.

**Equipo**

- María Paula Acosta Luque — mp.acosta1@uniandes.edu.co
- Andrés Torres — ga.torresc1@uniandes.edu.co
- Marlon Álvarez Álvarez — m.alvareza2@uniandes.edu.co
- David Geronimo Quiroga Torres — d.quirogat@uniandes.edu.co
- **Experto:** Andrés Ramírez — Gerente de mercadeo, Nodos — gerente.mercadeo1@interrapidisimo.com

**Problema**

Inter Rapidísimo cuenta con una "Torre de Direcciones" con coordenadas GPS,
pero **sin** información de tipo de inmueble ni horarios de operación. Esto
provoca:

1. Domiciliarios sin contexto del destino (casa, oficina, restaurante, bodega)
   hasta llegar.
2. Intentos de entrega en horarios inadecuados, generando **15-20% de fallos
   evitables** por horario o tipo incorrecto.

**Justificación**

El sistema completo:

1. Clasifica el tipo de inmueble con CNN/VLM a partir de fotos del domiciliario.
2. Asigna horarios según tipo, combinando reglas y Google Places API.
3. Alimenta automáticamente la Torre de Direcciones, enriqueciéndola desde el
   lanzamiento de Inter App.

**Objetivo general**

Desarrollar un sistema de clasificación automática de inmuebles mediante
redes neuronales convolucionales integrado con asignación de horarios de
operación, para enriquecer la Torre de Direcciones y optimizar entregas en
logística de última milla.

**Objetivos específicos**

1. Construir un dataset etiquetado en 12 categorías (casa, apartamento,
   oficina, restaurante, tienda, farmacia, banco, centro comercial, bodega,
   hospital, colegio, coworking) combinando fuentes públicas, Google Street
   View y captura local en Bogotá.
2. Desarrollar modelos de clasificación con transfer learning sobre
   MobileNetV2, ResNet50 y EfficientNetB0, comparando precisión y eficiencia.
3. Implementar la asignación de horarios por categoría combinando reglas
   predefinidas y consulta dinámica a Google Places API.
4. Evaluar el sistema con accuracy, precision, recall, F1-score y análisis de
   concordancia entre horarios asignados y reales.

> Esta API expone la inferencia del pipeline (clasificación + horario) que se
> integra con Inter App y con la Torre de Direcciones.
