"""Llamadas focalizadas al VLM. Una llamada = una tarea.

Dos funciones:
  * `extract_description(img)` → VISIÓN. Una descripción corta por imagen
    que un domiciliario pueda usar para reconocer el lugar. Incluye TODO lo
    visible: colores de fachada, letreros (texto literal), horarios escritos,
    pistas del entorno, números, comercios contiguos.
  * `parse_schedule(text)` → SOLO TEXTO. Toma la descripción acumulada y
    devuelve `{lunes: ["HH:MM-HH:MM"], ...}` o `None` si no hay horarios.

`consolidate_schedule(parsed)` aplica `DEFAULT_SCHEDULE` si vino `None`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI, AuthenticationError
from PIL import Image

import config
from images import pil_to_data_url

log = logging.getLogger(__name__)

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# Horario por defecto: L-V 8:00-17:00, Sábado 8:00-12:00, Domingo cerrado.
DEFAULT_SCHEDULE: dict[str, list[str]] = {
    "lunes":     ["08:00-17:00"],
    "martes":    ["08:00-17:00"],
    "miercoles": ["08:00-17:00"],
    "jueves":    ["08:00-17:00"],
    "viernes":   ["08:00-17:00"],
    "sabado":    ["08:00-12:00"],
    "domingo":   [],
}


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=f"{config.DESCRIBE_URL}/v1",
            api_key=config.RUNPOD_API_KEY,
        )
        log.info(
            "describe vLLM client init base_url=%s/v1 model=%s",
            config.DESCRIBE_URL, config.DESCRIBE_MODEL,
        )
    return _client


# ────────────────────────────────────────────────────────────────────
# DESCRIPCIÓN (visión, 1 llamada por imagen)
# Dos prompts según el rol de la imagen:
#   - main      → fachada del inmueble (visual primero).
#   - secondary → letreros y entorno (OCR primero).
# La consolidación de horario corre después con `parse_schedule` sobre
# las descripciones concatenadas.
# ────────────────────────────────────────────────────────────────────

DESCRIPTION_SYSTEM = (
    "Describes inmuebles para domiciliarios. Respondes en español, en una "
    "descripción corta (1-2 frases), factual, sin opiniones y sin inventar."
)

DESCRIPTION_MAIN_PROMPT = (
    "Esta es la foto PRINCIPAL del inmueble. Escribe UNA descripción corta "
    "(máx. 2 frases) de la fachada para que un domiciliario reconozca el lugar.\n\n"
    "Incluye SOLO lo que se vea claramente:\n"
    "- color de fachada, puerta, reja, portón;\n"
    "- número de pisos, balcones, antejardín, garaje;\n"
    "- número de placa si es visible.\n\n"
    "No describas comercios contiguos ni el entorno en esta imagen. "
    "No uses listas. No repitas instrucciones. "
    "Si la imagen no muestra nada útil, responde: sin información visible."
)

DESCRIPTION_SECONDARY_PROMPT = (
    "Tarea: transcribir el texto visible en la imagen y describir el "
    "entorno, para que un domiciliario ubique el inmueble.\n\n"
    "1. TEXTO: cita entre comillas y de forma LITERAL todo lo que se lea: "
    "letreros, avisos, horarios escritos (días y rangos de hora), nombres "
    "de comercios, números de placa, direcciones. No parafrasees: copia el "
    "texto tal cual aparece, respetando mayúsculas y signos.\n"
    "2. ENTORNO (si aplica): una frase corta con comercios contiguos, "
    "esquinas, parques o puntos de referencia visibles.\n\n"
    "Formato: prosa breve, máximo 3 frases. No uses listas con guiones.\n"
    "Solo si la imagen no contiene NINGÚN texto ni elemento útil, responde "
    "exactamente: sin información visible."
)


async def extract_description(pil_image: Image.Image, role: str = "secondary") -> str:
    """Una llamada al VLM. `role` ∈ {'main','secondary'} elige el prompt."""
    prompt = DESCRIPTION_MAIN_PROMPT if role == "main" else DESCRIPTION_SECONDARY_PROMPT
    image_url = pil_to_data_url(pil_image)
    messages = [
        {"role": "system", "content": DESCRIPTION_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    try:
        resp = await _get_client().chat.completions.create(
            model=config.DESCRIBE_MODEL,
            messages=messages,
            max_tokens=180,
            temperature=0.0,
            stop=["\n\n", "<end_of_turn>", "<start_of_turn>"],
            extra_body={"repetition_penalty": 1.2},
        )
    except AuthenticationError:
        log.exception("extract_description failed (401) role=%s", role)
        return ""
    except Exception:
        log.exception("extract_description failed role=%s", role)
        return ""
    raw = (resp.choices[0].message.content or "").strip()
    log.info(
        "vLLM description response: role=%s finish_reason=%s tokens=%s raw=%r",
        role,
        resp.choices[0].finish_reason,
        resp.usage.completion_tokens if resp.usage else None,
        raw[:400],
    )
    return raw


# ────────────────────────────────────────────────────────────────────
# PARSE HORARIO (solo texto, 1 llamada)
# ────────────────────────────────────────────────────────────────────

SCHEDULE_SYSTEM = (
    "Extraes horarios de atención desde texto. Respondes SOLO con JSON válido "
    "o con la palabra null. Nada más."
)

SCHEDULE_PROMPT = (
    "Del siguiente texto, extrae el horario de atención y devuélvelo como JSON "
    "con esta forma exacta (todas las claves presentes, en minúscula y sin "
    "tildes):\n\n"
    "{{\"lunes\":[\"HH:MM-HH:MM\"],\"martes\":[\"HH:MM-HH:MM\"],"
    "\"miercoles\":[\"HH:MM-HH:MM\"],\"jueves\":[\"HH:MM-HH:MM\"],"
    "\"viernes\":[\"HH:MM-HH:MM\"],\"sabado\":[\"HH:MM-HH:MM\"],"
    "\"domingo\":[]}}\n\n"
    "Reglas:\n"
    "- Horas en formato 24h (HH:MM).\n"
    "- Un día con varios turnos: [\"08:00-12:00\",\"14:00-18:00\"].\n"
    "- Un día cerrado: lista vacía [].\n"
    "- Si el texto NO menciona horarios de atención, responde EXACTAMENTE: null\n\n"
    "Texto:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Responde SOLO con el JSON o con null."
)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _validate_schedule_dict(obj: Any) -> dict[str, list[str]] | None:
    """Valida que sea {dia: [\"HH:MM-HH:MM\", ...]} con días canónicos."""
    if not isinstance(obj, dict):
        return None
    out: dict[str, list[str]] = {d: [] for d in DAYS}
    found_any = False
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        day = k.strip().lower()
        if day not in DAYS:
            continue
        if not isinstance(v, list):
            continue
        ranges: list[str] = []
        for r in v:
            if not isinstance(r, str):
                continue
            s = r.strip()
            if re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", s):
                ranges.append(s)
                found_any = True
        out[day] = ranges
    return out if found_any or any(d in obj for d in DAYS) else None


async def parse_schedule(description_text: str) -> dict[str, list[str]] | None:
    """Una llamada al LLM (solo texto). Devuelve dict válido o None."""
    text = (description_text or "").strip()
    if not text:
        return None

    messages = [
        {"role": "system", "content": SCHEDULE_SYSTEM},
        {"role": "user", "content": SCHEDULE_PROMPT.format(text=text)},
    ]
    try:
        resp = await _get_client().chat.completions.create(
            model=config.DESCRIBE_MODEL,
            messages=messages,
            max_tokens=220,
            temperature=0.0,
            stop=["<end_of_turn>", "<start_of_turn>"],
            extra_body={"repetition_penalty": 1.0},
        )
    except AuthenticationError:
        log.exception("parse_schedule failed (401)")
        return None
    except Exception:
        log.exception("parse_schedule failed")
        return None

    raw = (resp.choices[0].message.content or "").strip()
    log.info(
        "vLLM schedule response: finish_reason=%s tokens=%s raw=%r",
        resp.choices[0].finish_reason,
        resp.usage.completion_tokens if resp.usage else None,
        raw[:400],
    )

    if raw.lower().startswith("null"):
        return None

    candidate = raw
    if not candidate.startswith("{"):
        m = _JSON_OBJ_RE.search(candidate)
        if not m:
            return None
        candidate = m.group(0)

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        log.warning("parse_schedule: JSON inválido raw=%r", raw[:200])
        return None

    return _validate_schedule_dict(obj)


# ────────────────────────────────────────────────────────────────────
# CONSOLIDACIÓN
# ────────────────────────────────────────────────────────────────────

def consolidate_schedule(
    parsed: dict[str, list[str]] | None,
) -> dict[str, Any]:
    """Aplica `DEFAULT_SCHEDULE` si `parsed` es None."""
    if parsed is None:
        return {"hours": DEFAULT_SCHEDULE, "source": "default"}
    return {"hours": parsed, "source": "detected"}
