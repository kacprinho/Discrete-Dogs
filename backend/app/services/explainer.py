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


SYSTEM = (
    "You explain cross-disciplinary analogies to a smart non-specialist, e.g. a hackathon judge. "
    "You get a user's problem and one research mechanism retrieved for it. "
    "Use only the information provided; do not invent results, numbers or sources. "
    "Write plain English: short sentences, no jargon unless you explain it in a few words. "
    "Make the steps specific to the user's situation, not generic. "
    "If the analogy is weak, say so plainly in 'caveats'."
)

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence, max 20 words: how the other field solves the same problem.",
        },
        "connection": {
            "type": "string",
            "description": "2-3 sentences: what the shared underlying problem is, in plain words.",
        },
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 concrete things to try in the user's own system.",
        },
        "caveats": {
            "type": "string",
            "description": "One or two sentences on where the analogy may break down.",
        },
    },
    "required": ["headline", "connection", "steps", "caveats"],
    "additionalProperties": False,
}


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
    return (
        f"<user_problem>\n{problem.strip()[:2000]}\n</user_problem>\n\n"
        f"<retrieved_mechanism>\n{json.dumps(entry, ensure_ascii=False, indent=1)}\n</retrieved_mechanism>\n\n"
        "Explain how this mechanism could help with the user's problem."
    )


def _validate(text: str, model: str) -> Optional[Dict[str, Any]]:
    data = json.loads(text)
    steps = [s.strip() for s in data.get("steps", []) if isinstance(s, str) and s.strip()][:3]
    if not isinstance(data.get("headline"), str) or not data["headline"].strip() or not steps:
        return None
    return {
        "headline": data["headline"].strip(),
        "connection": str(data.get("connection", "")).strip(),
        "steps": steps,
        "caveats": str(data.get("caveats", "")).strip(),
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
