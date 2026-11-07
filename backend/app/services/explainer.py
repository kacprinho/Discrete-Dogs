"""Plain-English explanation layer: turns a retrieved match into a short, structured explanation.

Grounded: the model only sees the user's problem and the retrieved entry (title, summary,
mechanism, adaptation steps, limitations, citation) and is told to use nothing else.
Retrieval, scores and citations are never changed by this layer.

Provider (EXPLAINER_PROVIDER = auto | gemini | anthropic):
  auto -> Gemini if GEMINI_API_KEY is set, else Anthropic if ANTHROPIC_API_KEY is set, else off.
  Gemini: EXPLAINER_GEMINI_MODELS, tried in order (default gemini-3.5-flash-lite, then
  gemini-3.5-flash) so a 429/503 on one model falls through to the next.

Any failure (no key, timeout, refusal, bad JSON) returns None and the UI falls back to the
corpus text, so the demo never depends on it. Keys are read from the environment only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

try:
    import anthropic
except ImportError:  # optional dependency
    anthropic = None  # type: ignore[assignment]

logger = logging.getLogger("teleport.explainer")

ANTHROPIC_MODEL = os.environ.get("EXPLAINER_MODEL", "claude-haiku-5-5")
GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get("EXPLAINER_GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash").split(",")
    if m.strip()
]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_S = float(os.environ.get("EXPLAINER_TIMEOUT_S", "12"))


def provider() -> Optional[str]:
    if os.environ.get("EXPLAINER_ENABLED", "auto").lower() in ("0", "false", "off"):
        return None
    choice = os.environ.get("EXPLAINER_PROVIDER", "auto").lower()
    if choice in ("auto", "gemini") and os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if choice in ("auto", "anthropic") and anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def enabled() -> bool:
    return provider() is not None


# Jargon -> executive language. Used in the prompt AND applied to the output as a safety net.
JARGON_MAP: List["tuple[str, str]"] = [
    (r"scale-free (graph )?acquaintance (sampling|immunization)", "target high-impact hubs via their connected peers"),
    (r"acquaintance (sampling|immunization)", "finding key hubs through their connections"),
    (r"collaps\w* r[_ ]?eff\w*( below 1(\.0)?)?", "stop the spread before it multiplies"),
    (r"r[_ ]?eff(ective)?( reproduction number)?\s*(<|below)\s*1(\.0)?", "the spread stops multiplying"),
    (r"(two|2)-hop ring inoculation|ring inoculation", "isolating the immediate contact circle"),
    (r"(1|one|2|two)-hop (topological )?neighbou?rs?|topological neighbou?rs?", "immediate contact circle"),
    (r"verification gates?", "automatic safety checks"),
    (r"circuit[- ]breakers?", "automatic safety checks"),
    (r"dynamic velocity throttling( along edges)?", "slowing down how fast content moves"),
    (r"super-?spreaders?", "biggest spreaders"),
    (r"stigmerg\w*", "trail-following"),
    (r"allosteric", "shape-changing"),
    (r"exponential (volatile )?(decay|evaporation)", "gradual fading"),
    (r"kinematic viscosity", "built-in friction"),
    (r"\bviscosity\b", "friction"),
    (r"\btopolog(y|ies)\b", "network layout"),
    (r"\btopological\b", "structural"),
    (r"\bthermodynamics?\b", "heat-and-energy"),
    (r"\bthermodynamically\b", "energy-wise"),
    (r"\bstochastic(ally)?\b", "random"),
    (r"\beigenvectors?\b", "influence scores"),
    (r"\beigenvalues?\b", "key numbers"),
    (r"\bmarkov chains?\b", "step-by-step random process"),
    (r"\bmarkov\b", "step-by-step"),
    (r"\bentropy\b", "disorder"),
    (r"\btensors?\b", "multi-direction"),
    (r"\bgradients\b", "differences"),
    (r"\bgradient\b", "difference"),
    (r"epsilon-greedy", "occasional random"),
    (r"\bMVCC\b", "versioned records"),
    (r"\bp95\b", "normal worst-case"),
]

# (detect pattern in the source, jargon, executive translation) — only matching rows are sent.
JARGON_GUIDE: List["tuple[str, str, str]"] = [
    (r"acquaintance|scale-free", "Scale-free graph acquaintance sampling", "Target high-impact hubs via connected peers"),
    (r"r[_ ]?eff|reproduction number", "Collapse R_eff below 1.0", "Stop the spread before it multiplies"),
    (r"hop|ring inoculation|topological neighbou?r", "Topological neighbors / 2-hop ring inoculation", "Isolate the immediate contact circle"),
    (r"verification gate|circuit[- ]breaker", "Verification gates / circuit breakers", "Set up automatic safety checks"),
    (r"velocity throttl|throttl\w* along edges", "Dynamic velocity throttling along edges", "Slow down content/data transfer speed"),
    (r"stigmerg|pheromone", "Stigmergy / pheromone reinforcement", "Popular paths get marked, and the marks fade over time"),
    (r"allosteric", "Allosteric feedback", "The output itself tells the input to slow down"),
]


def _relevant_guide(entry_text: str) -> str:
    import re

    rows = [f"- {j} -> {t}" for pat, j, t in JARGON_GUIDE if re.search(pat, entry_text, re.IGNORECASE)]
    return "\n".join(rows)


SYSTEM = (
    "You turn a research analogy into an executive briefing a busy judge can read in 10 seconds. "
    "You get a user's problem and one research mechanism retrieved for it.\n"
    "Rules:\n"
    "- Use only the information provided; never invent results, numbers or sources.\n"
    "- Zero academic or technical jargon: no equations, Greek letters, model names, author names, "
    "or field-specific terms. If an idea needs a term, replace it with everyday words.\n"
    "- Short, direct, action-oriented sentences. Talk about the user's own system, not the paper.\n"
    "- Each step must be specific to the user's situation (their domain, their nouns).\n"
    "- If the analogy is weak, say so plainly in 'caveat'."
)

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "insight": {
            "type": "string",
            "description": "The core insight in ONE plain sentence (max 25 words) a non-expert understands instantly.",
        },
        "steps": {
            "type": "array",
            "description": "Exactly 3 steps of an execution plan for the user's own system, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "2-6 word imperative, e.g. 'Find the source'."},
                    "detail": {"type": "string", "description": "One plain sentence, max 20 words."},
                },
                "required": ["title", "detail"],
                "additionalProperties": False,
            },
        },
        "how_it_works": {
            "type": "string",
            "description": "1-2 plain sentences: how the other field (nature, physics, ...) solves the same problem.",
        },
        "caveat": {"type": "string", "description": "One plain sentence on where this may not apply."},
    },
    "required": ["insight", "steps", "how_it_works", "caveat"],
    "additionalProperties": False,
}


def dejargon(text: str) -> str:
    import re

    out = text
    for pattern, repl in JARGON_MAP:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", out).strip()


def _build_prompt(problem: str, match: Dict[str, Any]) -> str:
    entry = {
        "title": match.get("title"),
        "field": match.get("domain"),
        "summary": match.get("summary"),
        "mechanism": match.get("mechanism"),
        "adaptation_steps_from_source": match.get("adaptation") or [],
        "known_limitations": match.get("limitations") or [],
        "citation": match.get("paper_ref"),
    }
    guide = _relevant_guide(json.dumps(entry, ensure_ascii=False))
    guide_block = (
        "<jargon_translations>\nThe source uses these terms; if you need them, say them like this:\n"
        f"{guide}\n</jargon_translations>\n\n"
        if guide
        else ""
    )
    return (
        f"<user_problem>\n{problem.strip()[:2000]}\n</user_problem>\n\n"
        + guide_block
        + f"<retrieved_mechanism>\n{json.dumps(entry, ensure_ascii=False, indent=1)}\n</retrieved_mechanism>\n\n"
        "Write the executive briefing for this user's problem."
    )


def _validate(text: str, model: str) -> Optional[Dict[str, Any]]:
    data = json.loads(text)
    steps: List[Dict[str, str]] = []
    for st in data.get("steps", []) or []:
        if isinstance(st, dict) and str(st.get("title", "")).strip():
            steps.append({"title": dejargon(str(st["title"])), "detail": dejargon(str(st.get("detail", "")))})
        elif isinstance(st, str) and st.strip():
            steps.append({"title": dejargon(st), "detail": ""})
    insight = str(data.get("insight", "")).strip()
    if not insight or not steps:
        return None
    return {
        "insight": dejargon(insight),
        "steps": steps[:3],
        "how_it_works": dejargon(str(data.get("how_it_works", ""))),
        "caveat": dejargon(str(data.get("caveat", ""))),
        "model": model,
    }


# ---------------------------------------------------------------- Gemini (REST)

async def _gemini_call(model: str, prompt: str) -> httpx.Response:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
            "temperature": 0.3,
            "maxOutputTokens": 4096,  # includes thinking tokens on Gemini 3.x
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        return await client.post(
            GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},  # header, never the URL/logs
            json=body,
        )


async def _explain_gemini(prompt: str) -> Optional[Dict[str, Any]]:
    # Free-tier keys hit per-minute limits in bursts: one retry round after a short pause.
    for attempt in range(2):
        out, busy = await _gemini_round(prompt)
        if out is not None or not busy:
            return out
        if attempt == 0:
            await asyncio.sleep(1.5)
    return None


async def _gemini_round(prompt: str) -> "tuple[Optional[Dict[str, Any]], bool]":
    """Tries each model once. Returns (result, all_models_busy)."""
    busy = 0
    for model in GEMINI_MODELS:
        try:
            resp = await _gemini_call(model, prompt)
        except httpx.HTTPError as exc:
            logger.info("gemini %s transport error: %s", model, type(exc).__name__)
            busy += 1
            continue
        if resp.status_code in (429, 500, 503):
            logger.info("gemini %s busy (%s) — trying next model", model, resp.status_code)
            busy += 1
            continue
        if resp.status_code != 200:
            logger.info("gemini %s HTTP %s", model, resp.status_code)
            return None, False  # auth/request errors won't be fixed by another model
        data = resp.json()
        cand = (data.get("candidates") or [{}])[0]
        if cand.get("finishReason") not in (None, "STOP"):
            logger.info("gemini %s finishReason=%s", model, cand.get("finishReason"))
            continue
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if text:
            try:
                out = _validate(text, data.get("modelVersion") or model)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.info("gemini %s invalid JSON: %s", model, type(exc).__name__)
                continue  # e.g. truncated output: let the next model try
            if out is not None:
                return out, False
    return None, busy == len(GEMINI_MODELS)


# ---------------------------------------------------------------- Anthropic (SDK)

_anthropic_client: Any = None


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(timeout=TIMEOUT_S, max_retries=1)
    return _anthropic_client


async def _explain_anthropic(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        response = await _get_anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIConnectionError as exc:
        logger.info("anthropic connection error: %s", exc)
        return None
    except anthropic.APIStatusError as exc:
        logger.info("anthropic API error %s", exc.status_code)
        return None
    if response.stop_reason != "end_turn":
        logger.info("anthropic stop_reason=%s", response.stop_reason)
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    return _validate(text, ANTHROPIC_MODEL)


_PLAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "plain_cards.json")
_plain: Optional[Dict[str, Dict[str, Any]]] = None


def precomputed(doc_id: str) -> Optional[Dict[str, Any]]:
    """Pre-generated plain-English card for a corpus entry (see scripts/build_plain_cards.py)."""
    global _plain
    if _plain is None:
        try:
            with open(_PLAIN_PATH, encoding="utf-8") as fh:
                _plain = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _plain = {}
    card = _plain.get(doc_id)
    if not card:
        return None
    return {
        "insight": dejargon(card.get("insight", "")),
        "steps": [
            {"title": dejargon(st.get("title", "")), "detail": dejargon(st.get("detail", ""))}
            for st in card.get("steps", [])
            if isinstance(st, dict)
        ],
        "how_it_works": dejargon(card.get("how_it_works", "")),
        "caveat": dejargon(card.get("caveat", "")),
        "model": "pre-generated",
    }


_cache: Dict[str, Dict[str, Any]] = {}  # same problem + same match -> no repeat API call


async def explain(problem: str, match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    which = provider()
    if which is None or not problem.strip():
        return None
    prompt = _build_prompt(problem, match)
    key = hashlib.sha256(f"{which}|{prompt}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    try:
        out = await (_explain_gemini(prompt) if which == "gemini" else _explain_anthropic(prompt))
        if out is not None:
            if len(_cache) > 256:
                _cache.clear()
            _cache[key] = out
        return out
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.info("explainer bad output: %s", exc)
    except Exception as exc:  # never break the teleport stream
        logger.warning("explainer unexpected failure: %s", type(exc).__name__)
    return None
