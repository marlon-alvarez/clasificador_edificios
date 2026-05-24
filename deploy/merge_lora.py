"""
merge_lora.py — Mergea un adapter LoRA con su modelo base Gemma 3 4B
y opcionalmente lo sube a Hugging Face Hub como repo privado.

Uso:
    export HF_TOKEN="hf_xxx"
    huggingface-cli login --token $HF_TOKEN

    python merge_lora.py \
        --adapter_dir ./best_adapter \
        --output_dir  ./gemma-3-4b-ft-merged \
        --hf_repo_id  andtorrcan94/gemma-3-4b-ft-merged \
        --private

Notas importantes:
- Carga el modelo como Gemma3ForConditionalGeneration (multimodal completo)
  porque tu adapter toca tanto el LM como el vision tower (SigLIP).
- modules_to_save=[lm_head, embed_tokens] se persisten correctamente al
  llamar merge_and_unload() — esos módulos se guardan como pesos completos.
- bf16 para mantener calidad y reducir uso de VRAM/RAM.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True,
                        help="Carpeta con adapter_config.json + adapter_model.safetensors")
    parser.add_argument("--output_dir", required=True,
                        help="Donde guardar el modelo merged en disco")
    parser.add_argument("--hf_repo_id", default=None,
                        help="Si se proporciona, push a HF Hub (ej: usuario/modelo)")
    parser.add_argument("--private", action="store_true",
                        help="Crear el repo HF como privado")
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | cpu (cpu es lento pero funciona)")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leer base_model_name_or_path desde el adapter_config
    import json
    with open(adapter_dir / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg["base_model_name_or_path"]
    print(f"[1/5] Base model: {base_model_id}")

    # 2. Cargar base en bf16
    print(f"[2/5] Cargando base model en bfloat16 (puede tardar y descargar ~9GB)...")
    base = Gemma3ForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )

    # 3. Aplicar adapter LoRA
    print(f"[3/5] Cargando adapter LoRA desde {adapter_dir}...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    # 4. Merge: fusiona LoRA en los pesos base + persiste modules_to_save
    print(f"[4/5] Mergeando LoRA + modules_to_save (lm_head, embed_tokens)...")
    model = model.merge_and_unload(progressbar=True)

    # 5. Guardar a disco con safe_serialization (multi-shard)
    print(f"[5/5] Guardando modelo merged en {output_dir}...")
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Copiar processor + tokenizer + chat_template del adapter para que el
    # repo final sea autocontenido (el adapter ya trae los archivos correctos
    # del fine-tune)
    print("Copiando tokenizer + processor + chat_template...")
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_dir))
        processor.save_pretrained(str(output_dir))
    except Exception as e:
        print(f"  AutoProcessor falló ({e}); copiando archivos manualmente...")
        for fname in ["tokenizer.json", "tokenizer_config.json",
                      "processor_config.json", "chat_template.jinja"]:
            src = adapter_dir / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)

    # Asegurar que chat_template.jinja se copia (no siempre lo guarda processor)
    src_template = adapter_dir / "chat_template.jinja"
    dst_template = output_dir / "chat_template.jinja"
    if src_template.exists() and not dst_template.exists():
        shutil.copy2(src_template, dst_template)

    print(f"\n✓ Merge completo en: {output_dir}")
    print(f"  Tamaño total: {sum(p.stat().st_size for p in output_dir.glob('*')) / 1e9:.2f} GB")

    # 6. Push a HF Hub (opcional)
    if args.hf_repo_id:
        from huggingface_hub import HfApi, create_repo
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Define HF_TOKEN env var antes de hacer push")

        print(f"\nCreando repo {args.hf_repo_id} (private={args.private})...")
        create_repo(
            repo_id=args.hf_repo_id,
            private=args.private,
            exist_ok=True,
            token=token,
        )

        print(f"Subiendo carpeta {output_dir} -> {args.hf_repo_id}...")
        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.hf_repo_id,
            token=token,
            commit_message="Initial upload: Gemma 3 4B merged with LoRA fine-tune",
        )
        print(f"\n✓ Disponible en: https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
