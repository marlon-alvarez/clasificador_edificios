"""Llamadas focalizadas al VLM, una por tipo de extracción.

Tres funciones independientes (cada una con prompt mínimo):
  * `extract_schedule(img)`   → horario de atención
  * `describe_surroundings(img)` → descripción del entorno físico

Cada función hace UNA llamada al modelo. La idea es que el modelo
fine-tuned (`gemma-3-4b-ft`), que tiende a outputs cortos, pueda
responder bien si la tarea es chica y específica.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import AsyncOpenAI, AuthenticationError
from PIL import Image

import config
from colors import PALETTE
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
            base_url=f"{config.RUNPOD_URL.rstrip('/')}/v1",
            api_key=config.RUNPOD_API_KEY,
        )
    return _client


# ── Regex de direcciones (heredado para `extract_fields`) ──
ADDRESS_RE = re.compile(
    r"\b("
    r"(?:calle|cl|cll|carrera|cra|kr|krra|avenida|av|avda|"
    r"diagonal|dg|diag|transversal|tv|trans|autopista|circular|circunvalar)\.?"
    r"[\s\.\-#°ºoNn]*"
    r"\d{1,4}[a-z]?"
    r"(?:\s*(?:bis|sur|norte|este|oeste))?"
    r"(?:\s*(?:#|n[°ºo]\.?|no\.?)\s*\d{1,4}[a-z]?)?"
    r"(?:\s*-\s*\d{1,4})?"
    r")",
    re.IGNORECASE,
)


def extract_fields(text: str) -> dict[str, Any]:
    text_norm = " ".join(text.split())
    addresses = sorted({m.group(1).strip() for m in ADDRESS_RE.finditer(text_norm)})
    lower = text_norm.lower()
    color_mentions = sorted({c for c in PALETTE if re.search(rf"\b{c}\b", lower)})
    return {"addresses": addresses, "color_mentions": color_mentions}


# ────────────────────────────────────────────────────────────────────
# SCHEDULE
# ────────────────────────────────────────────────────────────────────

SCHEDULE_SYSTEM = (
    "Lees imágenes de letreros y describes el horario de atención que ves "
    "en una sola frase corta en español."
)

SCHEDULE_PROMPT = (
    "Mira la imagen y describe en UNA frase el horario de atención que se ve: "
    "qué días, en qué horas. Usa lenguaje natural, sin listas ni tablas.\n\n"
    "Ejemplos de frases válidas (NO copies estos, solo son referencia de estilo):\n"
    "- \"Lunes a viernes de 9 a 6, sábado de 10 a 2, domingo cerrado.\"\n"
    "- \"De lunes a jueves de 8 a 12 y de 2 a 6 de la tarde, viernes de 7am a 3pm.\"\n"
    "- \"Sábados y domingos de 10 a 4.\"\n\n"
    "Si la imagen NO muestra ningún horario, responde solo: NINGUNO"
)


# ── Regex y parser para el horario en lenguaje natural ──

_TIME_RE = re.compile(r"(\d{1,2})(?::(\d{2}))?")

_DAY_NORMALIZE = {
    "lunes": "lunes", "martes": "martes",
    "miercoles": "miercoles", "miércoles": "miercoles",
    "jueves": "jueves", "viernes": "viernes",
    "sabado": "sabado", "sábado": "sabado",
    "domingo": "domingo",
}

# Reconoce 4 tipos de tokens en orden de prioridad (alternancia regex):
#  1. Rango de días: "lunes a viernes"
#  2. Día suelto: "viernes"
#  3. Rango de horas: "8:00 a 12:00", "8-12", "7 a 3 pm"
#  4. Cierre: "cerrado", "cerrada", "closed"
_TOKEN_RE = re.compile(
    r"(?P<d1>lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
    r"\s+(?:a|al|hasta|-|–|—)\s+"
    r"(?P<d2>lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
    r"|"
    r"(?P<dsingle>lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
    r"|"
    r"(?P<t1>\d{1,2}(?::\d{2})?)\s*(?:am|pm)?\s*(?:a|al|hasta|-|–|—)\s*"
    r"(?P<t2>\d{1,2}(?::\d{2})?)\s*(?P<ampm>am|pm|a\.?m\.?|p\.?m\.?|de\s+la\s+(?:ma[ñn]ana|tarde|noche))?"
    r"|"
    r"(?P<closed>cerrad[oa]s?|closed)",
    re.IGNORECASE,
)


def _norm_time(raw: str) -> tuple[int, int] | None:
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    hi = int(m.group(1))
    mi = int(m.group(2)) if m.group(2) else 0
    if not (0 <= hi <= 23 and 0 <= mi <= 59):
        return None
    return hi, mi


def _normalize_range(open_str: str, close_str: str, ampm: str | None) -> str | None:
    """Convierte 12h→24h con heurística + sufijo AM/PM si vino explícito.

    Heurística: si no hay sufijo y la hora de cierre es ≤ la de apertura y ≤ 7,
    asume PM y suma 12 a la hora de cierre (caso típico: '7 a 3' = 07-15).
    """
    o = _norm_time(open_str)
    c = _norm_time(close_str)
    if not o or not c:
        return None
    oh, om = o
    ch, cm = c

    suffix = (ampm or "").lower().replace(".", "").replace(" ", "")
    is_pm = "pm" in suffix or "tarde" in suffix or "noche" in suffix
    is_am = "am" in suffix or "mañana" in suffix or "manana" in suffix
    if is_pm:
        # Cierre PM siempre. Apertura PM sólo si oh+12 < ch (caso 'de 2 a 6pm').
        # Para '7am a 3pm', oh=7 + 12 = 19 > ch=15 → mantenemos oh=7.
        if ch < 12:
            ch += 12
        if oh < 12 and (oh + 12) < ch:
            oh += 12
    elif is_am:
        if oh == 12:
            oh = 0
        if ch == 12:
            ch = 0
    elif (ch < oh or (ch == oh and cm <= om)) and ch <= 7:
        # sin sufijo y cierre menor que apertura: cierre asumido PM
        ch += 12

    if ch > 23 or oh > 23:
        return None
    return f"{oh:02d}:{om:02d}-{ch:02d}:{cm:02d}"


def _expand_day_range(d1: str, d2: str) -> list[str]:
    a = _DAY_NORMALIZE.get(d1.lower())
    b = _DAY_NORMALIZE.get(d2.lower())
    if not a or not b:
        return []
    i, j = DAYS.index(a), DAYS.index(b)
    if i <= j:
        return DAYS[i:j + 1]
    return DAYS[i:] + DAYS[:j + 1]


def _parse_schedule(raw: str) -> dict[str, list[str]] | None:
    """Parsea descripciones en lenguaje natural:
       'Lunes a jueves de 8 a 12 y de 2 a 6, viernes 7 a 3, sábado cerrado.'
    También acepta el formato estructurado 'LUNES: 08:00-12:00\\n...' como caso particular.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.upper().startswith("NINGUNO"):
        return None

    out: dict[str, list[str]] = {d: [] for d in DAYS}
    current_days: list[str] = []
    dirty = False  # True después de asociar un rango de tiempo a current_days
    found_any = False

    for m in _TOKEN_RE.finditer(cleaned):
        if m.group("d1") and m.group("d2"):
            current_days = _expand_day_range(m.group("d1"), m.group("d2"))
            dirty = False
        elif m.group("dsingle"):
            day = _DAY_NORMALIZE.get(m.group("dsingle").lower())
            if not day:
                continue
            if dirty or not current_days:
                current_days = [day]
                dirty = False
            else:
                current_days.append(day)
        elif m.group("t1") and m.group("t2"):
            r = _normalize_range(m.group("t1"), m.group("t2"), m.group("ampm"))
            if r and current_days:
                for d in current_days:
                    if r not in out[d]:
                        out[d].append(r)
                found_any = True
                dirty = True
        elif m.group("closed"):
            dirty = True  # los días previos quedan con su lista (vacía si no se llenó)

    if not found_any:
        return None
    return out


async def extract_schedule(pil_image: Image.Image) -> dict[str, list[str]] | None:
    """Una llamada al VLM enfocada en horario. Devuelve dict o None."""
    image_url = pil_to_data_url(pil_image)
    messages = [
        {"role": "system", "content": SCHEDULE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": SCHEDULE_PROMPT},
            ],
        },
    ]
    try:
        resp = await _get_client().chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=120,
            temperature=0.1,
            stop=["\n\n", "\nmodel", "<end_of_turn>", "<start_of_turn>"],
            extra_body={"repetition_penalty": 1.2},
        )
    except AuthenticationError:
        log.exception("extract_schedule failed (401)")
        return None
    except Exception:
        log.exception("extract_schedule failed")
        return None
    raw = (resp.choices[0].message.content or "").strip()
    log.info(
        "vLLM schedule response: finish_reason=%s tokens=%s raw=%r",
        resp.choices[0].finish_reason,
        resp.usage.completion_tokens if resp.usage else None,
        raw[:600],
    )
    return _parse_schedule(raw)


# ────────────────────────────────────────────────────────────────────
# SURROUNDINGS
# ────────────────────────────────────────────────────────────────────

SURROUNDINGS_SYSTEM = (
    "Eres un asistente que describe el entorno físico de inmuebles. "
    "Respondes con 1 o 2 frases en español, factual y conciso."
)

SURROUNDINGS_PROMPT = (
    "Describe en 1 o 2 frases el entorno físico visible alrededor del inmueble "
    "en la imagen: comercios contiguos, edificios vecinos, materiales o colores "
    "distintivos, hitos visuales. Útil para localizar la dirección.\n\n"
    "Si la imagen NO permite describir el entorno (porque muestra solo un "
    "letrero, un interior, una foto cerrada, etc.), responde EXACTAMENTE: "
    "sin referencias visibles\n\n"
    "No repitas instrucciones. No uses listas. Solo la descripción."
)


async def describe_surroundings(pil_image: Image.Image) -> str:
    """Una llamada al VLM enfocada en descripción del entorno. Devuelve string."""
    image_url = pil_to_data_url(pil_image)
    messages = [
        {"role": "system", "content": SURROUNDINGS_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": SURROUNDINGS_PROMPT},
            ],
        },
    ]
    try:
        resp = await _get_client().chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=100,
            temperature=0.3,
            stop=["\n\n", "\nmodel", "<end_of_turn>", "<start_of_turn>"],
            extra_body={"repetition_penalty": 1.3},
        )
    except AuthenticationError:
        log.exception("describe_surroundings failed (401)")
        return ""
    except Exception:
        log.exception("describe_surroundings failed")
        return ""
    raw = (resp.choices[0].message.content or "").strip()
    log.info(
        "vLLM surroundings response: finish_reason=%s tokens=%s raw=%r",
        resp.choices[0].finish_reason,
        resp.usage.completion_tokens if resp.usage else None,
        raw[:400],
    )
    return raw


# ────────────────────────────────────────────────────────────────────
# CONSOLIDACIÓN
# ────────────────────────────────────────────────────────────────────

def consolidate_schedule(
    per_image_schedules: list[dict[str, list[str]] | None],
    base_index: int = 1,
) -> dict[str, Any]:
    """Primer horario detectado entre las imágenes; default si ninguno."""
    for i, sched in enumerate(per_image_schedules):
        if sched:
            return {"hours": sched, "source": "detected", "image_index": base_index + i}
    return {"hours": DEFAULT_SCHEDULE, "source": "default", "image_index": None}
