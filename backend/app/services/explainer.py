"""Plain-English explanation layer: turns a retrieved match into a short, structured explanation.

Grounded: the model only sees the user's problem and the retrieved entry (title, summary,
mechanism, adaptation steps, limitations, citation) and is told to use nothing else.
Retrieval, scores and citations are never changed by this layer.

Enabled when ANTHROPIC_API_KEY is set (or EXPLAINER_ENABLED=1 with another credential source).
Any failure (no key, timeout, refusal, bad JSON) returns None and the UI falls back to the
corpus text, so the demo never depends on it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

try:
    import anthropic
except ImportError:  # optional dependency: the explainer simply stays off
    anthropic = None  # type: ignore[assignment]

logger = logging.getLogger("teleport.explainer")

MODEL = os.environ.get("EXPLAINER_MODEL", "claude-haiku-5-5")  # $0.10 / $0.50 per MTok
TIMEOUT_S = float(os.environ.get("EXPLAINER_TIMEOUT_S", "12"))


def enabled() -> bool:
    if anthropic is None:
        return False
    flag = os.environ.get("EXPLAINER_ENABLED", "auto").lower()
    if flag in ("0", "false", "off"):
        return False
    if flag in ("1", "true", "on"):
        return True
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


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

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(timeout=TIMEOUT_S, max_retries=1)
    return _client


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


async def explain(problem: str, match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not enabled() or not problem.strip():
        return None
    try:
        client = _get_client()
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": _build_prompt(problem, match)}],
        )
        if response.stop_reason != "end_turn":
            logger.info("explainer stop_reason=%s", response.stop_reason)
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        steps = [s.strip() for s in data.get("steps", []) if isinstance(s, str) and s.strip()][:3]
        if not data.get("headline") or not steps:
            return None
        return {
            "headline": data["headline"].strip(),
            "connection": data.get("connection", "").strip(),
            "steps": steps,
            "caveats": data.get("caveats", "").strip(),
            "model": MODEL,
        }
    except anthropic.APIConnectionError as exc:
        logger.info("explainer connection error: %s", exc)
    except anthropic.RateLimitError:
        logger.info("explainer rate limited")
    except anthropic.APIStatusError as exc:
        logger.info("explainer API error %s: %s", exc.status_code, exc.message)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.info("explainer bad output: %s", exc)
    except Exception as exc:  # never break the teleport stream
        logger.warning("explainer unexpected failure: %s", exc)
    return None
