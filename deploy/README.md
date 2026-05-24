# Deploy — Merge LoRA → Hugging Face

Esta carpeta empaqueta el adapter LoRA del clasificador (`best_adapter/`) con
el modelo base **Gemma 3 4B IT** y publica el resultado mergeado en un repo
privado de Hugging Face. Ese repo es el que luego sirve vLLM en el pod GPU
(ver `../api/PLAN_DEPLOY_RUNPOD.md`).

## Contenido

```
deploy/
  best_adapter/        # adapter LoRA fine-tuned (≈2.7 GB) + tokenizer/processor
  merge_lora.py        # carga base + adapter, mergea, guarda y sube a HF
  setup_cpu_pod.sh     # script "todo-en-uno" para correr el merge en un pod CPU
  requirements.txt     # transformers, peft, accelerate, huggingface_hub, torch
```

## Flujo

1. Levantar un **pod CPU** en RunPod (no hace falta GPU para el merge; CPU
   tarda ~10–15 min y evita pagar GPU mientras solo copiás pesos).
2. Subir esta carpeta `deploy/` a `/workspace` del pod (vía `runpodctl` o
   Jupyter).
3. Setear env vars en el pod:
   - `HF_TOKEN` — token de Hugging Face con permisos **read+write**.
   - `HF_REPO_ID` — destino, ej. `andtorrcan94/gemma-3-4b-ft-merged`.
4. En el Web Terminal del pod:
   ```bash
   cd /workspace
   bash setup_cpu_pod.sh
   ```
5. Cuando termine, **terminar el pod CPU** y crear el pod GPU que va a servir
   el modelo con vLLM apuntando al `HF_REPO_ID` recién creado (ver
   [Despliegue del pod GPU](#despliegue-del-pod-gpu)).

## Despliegue del pod GPU

Para servir el modelo mergeado usamos la imagen oficial
**`vllm/vllm-openai:v0.21.0`** directamente desde el template de RunPod, sin
construir una imagen propia. La elegimos porque:

- Expone un servidor **compatible con la API de OpenAI** sobre vLLM, así el
  cliente en [`api/classifier.py`](../api/classifier.py) y
  [`api/extractor.py`](../api/extractor.py) habla con el pod usando el SDK
  oficial de OpenAI sin adaptadores.
- Viene optimizada con **CUDA 12.9 sobre Ubuntu 24.04**, lista para GPUs
  modernas (Ada/Hopper) sin tocar drivers.
- Incluye soporte nativo para **modelos multimodales** (necesario para
  Gemma 3 4B con vision tower SigLIP) y para **`guided_choice`**, que es lo
  que fuerza al clasificador a responder exactamente una de las clases
  (`apartamento | casa | local_comercial | unknown`).

Por eso desplegar el fine-tune en RunPod se reduce a levantar dos procesos
vLLM en el mismo pod GPU apuntando a la imagen oficial, uno por modelo:

| Proceso | Puerto | Modelo (`--model`) | `--served-model-name` |
|---|---|---|---|
| Clasificador (fine-tune) | `8000` | `andtorrcan94/gemma-3-4b-ft-merged` | `gemma-3-4b-ft` |
| Base (descripción/OCR/horario) | `8001` | `google/gemma-3-4b-it` | `gemma-3-4b-it` |

Ambos procesos comparten la misma `--api-key`, que es la que la API local
expone como `RUNPOD_API_KEY` en su `.env`.

## Qué hace `merge_lora.py`

1. Lee `base_model_name_or_path` del `adapter_config.json` (Gemma 3 4B IT).
2. Carga la base en **bf16** como `Gemma3ForConditionalGeneration` (multimodal
   completo: LM + vision tower SigLIP) — necesario porque el adapter toca
   ambas torres.
3. Aplica el adapter LoRA con `PeftModel.from_pretrained` y llama
   `merge_and_unload()`, que además persiste los `modules_to_save`
   (`lm_head`, `embed_tokens`) como pesos completos.
4. Guarda el modelo mergeado con `safe_serialization` y `max_shard_size=4GB`.
5. Copia tokenizer + processor + `chat_template.jinja` del adapter para que el
   repo final sea autocontenido.
6. (Opcional) Crea el repo en HF Hub y sube la carpeta.

## Uso manual (sin `setup_cpu_pod.sh`)

```bash
pip install -r requirements.txt
export HF_TOKEN="hf_xxx"
huggingface-cli login --token "$HF_TOKEN"

python merge_lora.py \
  --adapter_dir ./best_adapter \
  --output_dir  ./gemma-3-4b-ft-merged \
  --hf_repo_id  andtorrcan94/gemma-3-4b-ft-merged \
  --private \
  --device cpu       # auto | cuda | cpu
```

## Notas

- El adapter pesa ~2.7 GB y el modelo mergeado resultante ~9 GB; asegurate de
  tener al menos ~25 GB libres en `/workspace` para la base descargada + el
  merge + el push.
- `--device cpu` funciona pero es lento; en CPU el cuello de botella es la
  carga del base model desde HF (~9 GB de download). Si ya está cacheado, el
  merge en sí es minutos.
- El repo de destino se crea como **privado**; revocá el token después si no
  vas a re-publicar.
