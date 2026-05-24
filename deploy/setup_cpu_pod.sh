#!/usr/bin/env bash
# setup_cpu_pod.sh — Pega este script completo en el Web Terminal del Pod CPU
# de RunPod para hacer el merge del LoRA + push a Hugging Face.
#
# Requiere que el Pod tenga estas env vars seteadas en su config:
#   HF_TOKEN     = tu token de Hugging Face con read+write
#   HF_REPO_ID   = ej. andtorrcan94/gemma-3-4b-ft-merged
#
# Y que la carpeta ./best_adapter ya esté en /workspace (subida vía
# runpodctl o Jupyter).

set -e

cd /workspace

echo "==> 1/4 Verificando archivos del adapter..."
if [ ! -f "best_adapter/adapter_config.json" ]; then
  echo "ERROR: no encuentro best_adapter/adapter_config.json en /workspace"
  echo "Sube la carpeta best_adapter antes de correr este script."
  exit 1
fi

echo "==> 2/4 Instalando dependencias..."
pip install --quiet --upgrade \
  "transformers>=4.50" \
  "peft>=0.19" \
  "accelerate>=0.34" \
  "huggingface_hub>=0.25" \
  safetensors sentencepiece

echo "==> 3/4 Autenticando en Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN no está seteada. Configúrala en las env vars del Pod."
  exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

echo "==> 4/4 Ejecutando merge_lora.py (esto tarda ~10-15 min)..."
if [ ! -f "merge_lora.py" ]; then
  echo "merge_lora.py no encontrado en /workspace. Descárgalo o pégalo manualmente."
  exit 1
fi

python merge_lora.py \
  --adapter_dir ./best_adapter \
  --output_dir  ./gemma-3-4b-ft-merged \
  --hf_repo_id  "${HF_REPO_ID:-andtorrcan94/gemma-3-4b-ft-merged}" \
  --private \
  --device cpu

echo ""
echo "================================================================"
echo "✓ Listo. Modelo merged y subido a:"
echo "  https://huggingface.co/${HF_REPO_ID:-andtorrcan94/gemma-3-4b-ft-merged}"
echo ""
echo "Siguiente paso: TERMINA este Pod CPU y crea el Pod GPU para servir."
echo "================================================================"
