# Plan de Despliegue — API Clasificador de Edificios en RunPod

**Versión:** 2.0
**Fecha:** 2026-05-18
**Objetivo:** Desplegar la API FastAPI en un pod RunPod (CPU es suficiente) conectada a un pod vLLM separado que sirve un modelo Gemma-3-VL fine-tuned para clasificación y extracción de metadatos a partir de fotos de inmuebles.

---

## 1. Resumen Ejecutivo

La API expone dos endpoints:

- `GET /health` — estado, modelo, URL vLLM, clases disponibles.
- `POST /processing` — recibe N imágenes (`multipart/form-data`, campo `images`) y devuelve clasificación + descripción por imagen + horario consolidado.

**Pipeline interno** ([api/main.py](api/main.py)):

1. Carga PIL de cada imagen (`load_pil`).
2. **Anonimización local de caras** con OpenCV Haar cascades — sin modelos externos, todo CPU ([api/anonymizer.py](api/anonymizer.py)).
3. **En paralelo** contra vLLM:
   - 1 llamada VLM para clasificar la imagen principal (índice 0) → `apartamento | casa | local_comercial`.
   - 1 llamada VLM por imagen para extraer una descripción corta orientada a domiciliarios (fachada, letreros, comercios contiguos, horarios escritos).
4. **1 llamada LLM solo-texto** sobre las descripciones concatenadas → JSON de horario semanal o `null`. Si `null`, aplica `DEFAULT_SCHEDULE` (L-V 08-17, Sáb 08-12, Dom cerrado).
5. Respuesta JSON estructurada.

**Dependencia dura:** un **pod vLLM** ya corriendo y accesible vía HTTPS. Sin él toda llamada de visión/texto devuelve `unknown` / horario default.

---

## 2. Requisitos

### Pod RunPod (esta API)
- **Tipo:** CPU pod (no requiere GPU). Cualquier preset económico sirve.
- **vCPU:** 1–2.
- **RAM:** 2 GB mínimo, 4 GB recomendado (OpenCV + PIL + buffers de imagen).
- **Disco:** 5–10 GB (OS + venv + deps).
- **Imagen base:** Ubuntu 22.04 (template "RunPod Pytorch" funciona pero es excesivo; `runpod/base:0.4.0-cpu` o cualquier Ubuntu 22 es suficiente).
- **Puertos expuestos:** TCP `8080` (HTTP) en el panel de RunPod para que aparezca como `https://<POD-ID>-8080.proxy.runpod.net`.

### Pod vLLM (precondición, ya debe existir)
- GPU (típicamente A6000 / L40 / A100 según latencia deseada).
- vLLM levantado con `--api-key <SECRETO>` y `--served-model-name gemma-3-4b-ft` (o el nombre que use `MODEL_NAME`).
- Puerto `8000` expuesto vía proxy HTTPS de RunPod.

### Software en el pod de la API
- Python **3.10+** (3.11 recomendado).
- Paquetes de sistema mínimos para `opencv-python-headless`: ya vienen en la wheel oficial; **no hace falta** instalar `libopencv-dev` ni `python3-opencv` del sistema.

### Credenciales necesarias
| Variable | Descripción |
|---|---|
| `RUNPOD_URL` | URL pública del pod vLLM, sin `/v1` ni trailing slash. Ej: `https://abc123-8000.proxy.runpod.net` |
| `RUNPOD_API_KEY` | Misma clave pasada a vLLM con `--api-key`. |
| `MODEL_NAME` | Nombre del modelo servido. Default: `gemma-3-4b-ft`. |
| `CLASS_NAMES` | Lista separada por coma. Default: `apartamento,casa,local_comercial`. |
| `PORT` | Puerto HTTP de la API. Default: `8080`. |

Fuente de verdad: [api/config.py](api/config.py) y [api/.env.example](api/.env.example).

---

## 3. Estructura del Proyecto

```
clasificador_edificios/
├── api/
│   ├── main.py          # FastAPI app + /health + /processing
│   ├── config.py        # Carga de .env y constantes
│   ├── anonymizer.py    # OpenCV Haar — blur de caras (CPU)
│   ├── classifier.py    # 1 llamada VLM → label
│   ├── extractor.py     # descripción por imagen + parse de horario
│   ├── images.py        # load_pil + pil_to_data_url
│   ├── requirements.txt # deps mínimas
│   ├── .env.example
│   └── README.md
└── plan.md              # este archivo
```

### `requirements.txt` actual ([api/requirements.txt](api/requirements.txt))
```
fastapi>=0.110
uvicorn[standard]>=0.27
python-dotenv>=1.0.0
python-multipart>=0.0.9
openai>=1.40
pillow>=10.0
numpy>=1.26
opencv-python-headless>=4.9
```

> **Nota:** versiones anteriores del plan mencionaban `mediapipe` y `easyocr`. Ya no se usan: la anonimización es Haar (CPU, sin descargas) y el OCR/descripción lo hace directamente el VLM.

---

## 4. Despliegue paso a paso

### 4.1 Crear y arrancar el pod en RunPod

1. RunPod Console → **Deploy** → **CPU Pods** (o GPU si ya pagás uno y querés reusar).
2. Selecciona un preset 1–2 vCPU / 4 GB RAM / 10 GB disco.
3. En **Expose HTTP Ports** agrega `8080`.
4. Imagen: `runpod/base:0.4.0-cpu` (o cualquier Ubuntu 22). Si usás un template con Pytorch va a funcionar igual.
5. **Deploy** y espera estado `Running`.

### 4.2 Conectarse

```bash
# Desde tu máquina
ssh root@<POD-ID>.proxy.runpod.net   # o usa "Web Terminal" desde la UI
```

### 4.3 Instalar Python y clonar el repo

```bash
apt update && apt install -y python3.11 python3.11-venv python3-pip git
cd /root
git clone https://github.com/<usuario>/clasificador_edificios.git
cd clasificador_edificios/api
```

Si el repo es privado, usar token HTTPS:
```bash
git clone https://<GITHUB_TOKEN>@github.com/<usuario>/clasificador_edificios.git
```

### 4.4 Crear venv e instalar deps

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Tiempo estimado: 1–3 min.

### 4.5 Configurar `.env`

```bash
cp .env.example .env
nano .env
```

Contenido (sin comillas, sin espacios):
```
RUNPOD_URL=https://<VLLM-POD-ID>-8000.proxy.runpod.net
RUNPOD_API_KEY=<misma-clave-que-vllm>
MODEL_NAME=gemma-3-4b-ft
CLASS_NAMES=apartamento,casa,local_comercial
PORT=8080
```

### 4.6 Verificar conectividad al pod vLLM (obligatorio antes de arrancar)

```bash
curl -sS "$RUNPOD_URL/v1/models" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python -m json.tool
```

Debe listar el modelo en `MODEL_NAME`. Si devuelve `401` o `Connection refused`, **detenerse**:
- `401` → revisar que `RUNPOD_API_KEY` coincida con la usada en `--api-key` de vLLM.
- `Connection refused` / DNS → revisar URL y que el pod vLLM esté `Running`.

### 4.7 Arranque manual (smoke test)

```bash
source .venv/bin/activate
python main.py
# INFO:     Uvicorn running on http://0.0.0.0:8080
```

En otra terminal:
```bash
curl -s http://localhost:8080/health | python -m json.tool
```

Esperado:
```json
{
  "status": "ok",
  "model": "gemma-3-4b-ft",
  "runpod_url": "https://...",
  "classes": ["apartamento","casa","local_comercial"]
}
```

### 4.8 Daemonizar con systemd

```bash
cat > /etc/systemd/system/api-edificios.service <<'EOF'
[Unit]
Description=Real Estate Image Processing API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/clasificador_edificios/api
EnvironmentFile=/root/clasificador_edificios/api/.env
ExecStart=/root/clasificador_edificios/api/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080 --workers 2
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now api-edificios
systemctl status api-edificios
journalctl -u api-edificios -f
```

> Si el pod no tiene systemd (algunos templates de RunPod corren sólo `bash`), usar alternativa con `tmux` o `nohup`:
> ```bash
> tmux new -s api 'cd /root/clasificador_edificios/api && source .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8080 --workers 2'
> ```

---

## 5. Contrato de la API

### `GET /health`
```json
{
  "status": "ok",
  "model": "gemma-3-4b-ft",
  "runpod_url": "https://...",
  "classes": ["apartamento","casa","local_comercial"]
}
```

### `POST /processing`

**Request:** `multipart/form-data`, campo `images` (uno o más archivos). La **primera** imagen es la principal del inmueble.

```bash
curl -X POST \
  -F "images=@principal.jpg" \
  -F "images=@letrero.jpg" \
  -F "images=@fachada_lateral.jpg" \
  https://<API-POD-ID>-8080.proxy.runpod.net/processing | python -m json.tool
```

**Response** (esquema actual, ver [api/main.py:112-123](api/main.py)):
```json
{
  "property_type": "casa",
  "classification": {
    "label": "casa",
    "raw": "casa",
    "error": null
  },
  "images": [
    {
      "index": 0,
      "filename": "principal.jpg",
      "role": "principal",
      "faces_blurred": 0,
      "description": "Casa de dos pisos color blanco con reja negra..."
    },
    {
      "index": 1,
      "filename": "letrero.jpg",
      "role": "secundaria",
      "faces_blurred": 1,
      "description": "Letrero 'Panadería La Esquina' horario 7:00-19:00..."
    }
  ],
  "schedule": {
    "hours": {
      "lunes": ["07:00-19:00"],
      "martes": ["07:00-19:00"],
      "miercoles": ["07:00-19:00"],
      "jueves": ["07:00-19:00"],
      "viernes": ["07:00-19:00"],
      "sabado": ["07:00-19:00"],
      "domingo": []
    },
    "source": "detected"
  },
  "summary": {
    "total_images": 2,
    "total_faces_blurred": 1,
    "schedule_source": "detected",
    "description_concatenated": "Casa de... | Letrero..."
  }
}
```

`schedule.source` ∈ `{"detected", "default"}`.

---

## 6. Latencia esperada

| Etapa | Costo |
|---|---|
| Carga de procesos en frío | ~1–3 s (no hay modelos a descargar) |
| Anonimización Haar por imagen | ~50–200 ms |
| `classify_property` (1 llamada VLM) | depende del pod vLLM, típicamente 1–4 s |
| `extract_description` por imagen | 1–4 s (en paralelo con clasificación) |
| `parse_schedule` (texto) | 0.5–2 s |
| **Total con 3 imágenes** | **~3–6 s** en caliente |

La concurrencia es importante: las llamadas a vLLM van con `asyncio.gather` ([main.py:88-94](api/main.py)).

---

## 7. Seguridad

1. `.env` **NO** se commitea (verificar `.gitignore` en `api/`).
2. `RUNPOD_API_KEY` debe ser un secreto fuerte; rotarlo cuando se reinicia el pod vLLM.
3. ⚠️ **TODO antes de producción:** quitar el log `[TESTING]` en [api/classifier.py:24-29](api/classifier.py) que imprime `RUNPOD_API_KEY` en cleartext. Reemplazar por log enmascarado:
   ```python
   masked = config.RUNPOD_API_KEY[:4] + "…" + config.RUNPOD_API_KEY[-4:]
   log.info("vLLM client init base=%s model=%s key=%s", base, config.MODEL_NAME, masked)
   ```
4. El proxy `*.proxy.runpod.net` ya termina TLS. Si se expone la API a clientes externos, agregar:
   - Auth simple por header (`X-API-Key`) en un middleware FastAPI.
   - Rate limiting con `slowapi` (5–10 req/min por IP).
5. Validar `Content-Length` / tamaño máximo de upload (uvicorn default permite multipart grande): considerar `--limit-max-requests` o un middleware que rechace > 20 MB por archivo.

---

## 8. Monitoreo y troubleshooting

### Logs
```bash
journalctl -u api-edificios -n 200 --no-pager
journalctl -u api-edificios -f
```

### Tabla de problemas comunes

| Síntoma | Causa probable | Acción |
|---|---|---|
| `/processing` devuelve `"label": "unknown"` y `error` en `classification` | API key incorrecta o vLLM no responde | `curl` a `$RUNPOD_URL/v1/models`. Revisar `.env` |
| `Connection refused` en logs | Pod vLLM caído / URL cambió tras restart | Renovar `RUNPOD_URL` en `.env` y `systemctl restart api-edificios` |
| `AuthenticationError` (401) repetido | Mismatch de `RUNPOD_API_KEY` | Re-copiar exactamente el secreto. Sin comillas. |
| `cv2.error` al cargar imagen | Imagen corrupta o no es imagen | El endpoint ya devuelve `400`; revisar input del cliente |
| OOM / pod se reinicia solo | Demasiadas imágenes grandes en paralelo | Subir RAM del pod o limitar `N` imágenes por request |
| `Address already in use` | Otro proceso en 8080 | `lsof -i:8080` y matar, o cambiar `PORT` en `.env` |
| Schedule siempre `"source": "default"` | El VLM no detecta horarios escritos en las fotos | Comportamiento esperado si no hay letrero con horario |

### Verificación rápida del entorno

```bash
source /root/clasificador_edificios/api/.venv/bin/activate
python -c "import fastapi, uvicorn, cv2, numpy, PIL, openai, dotenv; print('OK')"
```

---

## 9. Escalabilidad

- **Workers uvicorn:** `--workers 2` con 2 vCPU. No subir más allá de `vCPUs` físicas.
- **Cuello de botella real:** el pod vLLM, no esta API. Si necesitás throughput → escalar el pod GPU (más memoria → más concurrencia en vLLM) antes que esta capa.
- **Concurrencia por request:** `asyncio.gather` ya paraleliza clasificación + descripciones, así que un request con 10 imágenes son ~11 llamadas concurrentes a vLLM. Si vLLM se satura, considerar un semáforo (`asyncio.Semaphore`) en [extractor.py](api/extractor.py).
- **Sin estado:** la API es stateless, se puede correr varias réplicas detrás de un balanceador apuntando todas al mismo pod vLLM.

---

## 10. Checklist de despliegue

- [ ] Pod vLLM corriendo y `/v1/models` responde con auth.
- [ ] Pod API creado en RunPod con puerto `8080` expuesto.
- [ ] Python 3.10+ y `git` instalados.
- [ ] Repo clonado en `/root/clasificador_edificios`.
- [ ] `.venv` creado y `pip install -r requirements.txt` exitoso.
- [ ] `.env` con `RUNPOD_URL`, `RUNPOD_API_KEY`, `MODEL_NAME` válidos.
- [ ] `curl $RUNPOD_URL/v1/models` desde el pod responde 200.
- [ ] `python main.py` arranca sin error.
- [ ] `GET /health` desde fuera del pod responde 200 vía `https://<POD-ID>-8080.proxy.runpod.net/health`.
- [ ] `POST /processing` con imágenes reales devuelve `property_type` correcto.
- [ ] Service systemd activo (`systemctl is-active api-edificios` → `active`).
- [ ] Log `[TESTING]` de la API key removido o enmascarado.
- [ ] `.env` fuera de git (verificar con `git status`).

---

## 11. Próximos pasos (opcionales)

- Dockerfile + `docker build && docker push` a un registry, para reemplazar `git clone` por `docker pull` en el pod.
- GitHub Actions: lint + tests + build imagen al hacer merge a `main`.
- Endpoint `/metrics` (Prometheus) con `prometheus-fastapi-instrumentator`.
- Auth por API key + rate limiting (`slowapi`).
- Tests de integración con imágenes fixture + mock de vLLM.
- Considerar reemplazar Haar por un detector más robusto (RetinaFace / YuNet de OpenCV DNN) si la tasa de caras perdidas es alta.

---

**Estado:** Plan alineado con el código actual ([api/main.py](api/main.py) v2.0.0, deps de [api/requirements.txt](api/requirements.txt)).
