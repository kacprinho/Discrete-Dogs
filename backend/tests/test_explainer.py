"""Offline tests for the LLM explanation layer (Anthropic client is mocked)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.services import explainer

MATCH = {
    "title": "Ant Colony Pheromone Trail Reinforcement",
    "domain": "Natural & Biological Systems",
    "summary": "Ants balance routes with evaporating trails.",
    "mechanism": "positive feedback with decay",
    "adaptation": ["Weight routes by recent success"],
    "limitations": ["Needs warm-up traffic"],
    "paper_ref": "Bonabeau et al., Nature 2000",
}
GOOD = {
    "insight": "Ants spread traffic by letting busy trails fade.",
    "steps": [
        {"title": "Score each route", "detail": "Raise a route's score when requests succeed."},
        {"title": "Let scores fade", "detail": "Lower every score a little each second."},
        {"title": "Try quiet routes", "detail": "Send a few requests down idle paths."},
    ],
    "how_it_works": "Ant trails evaporate, so crowded paths lose their pull.",
    "caveat": "Needs enough traffic to learn.",
}


class FakeMessages:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        stop, text = self.behaviour
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def _install(monkeypatch, behaviour):
    fake = FakeMessages(behaviour)
    explainer._cache.clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(explainer, "_anthropic_client", SimpleNamespace(messages=fake))
    return fake


def test_success_returns_structured_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, ("end_turn", json.dumps(GOOD)))
    out = asyncio.run(explainer.explain("Traffic piles onto a few routes", MATCH))
    assert out and out["insight"] == GOOD["insight"] and len(out["steps"]) == 3
    assert out["steps"][0]["title"] == "Score each route"
    assert fake.kwargs["model"] == "claude-haiku-5-5"
    assert fake.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "Bonabeau" in fake.kwargs["messages"][0]["content"]  # grounded on the retrieved entry


@pytest.mark.parametrize(
    "behaviour",
    [
        ("refusal", ""),
        ("end_turn", "not json"),
        ("end_turn", json.dumps({**GOOD, "steps": []})),
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")),
    ],
)
def test_failures_fall_back_to_none(monkeypatch: pytest.MonkeyPatch, behaviour) -> None:
    _install(monkeypatch, behaviour)
    assert asyncio.run(explainer.explain("Traffic piles onto a few routes", MATCH)) is None


def test_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("EXPLAINER_ENABLED", raising=False)
    assert explainer.enabled() is False
    assert asyncio.run(explainer.explain("x", MATCH)) is None


# ------------------------------------------------------------------ Gemini provider

async def _no_sleep(_s):
    return None


def _gemini_resp(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "https://example"))


def _gemini_ok(text: str, finish: str = "STOP") -> dict:
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}],
            "modelVersion": "gemini-test"}


def _use_gemini(monkeypatch, responses):
    calls = []
    explainer._cache.clear()
    monkeypatch.setattr(explainer.asyncio, "sleep", _no_sleep)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async def fake_call(model, prompt):
        calls.append(model)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(explainer, "_gemini_call", fake_call)
    return calls


def test_gemini_preferred_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    calls = _use_gemini(monkeypatch, [_gemini_resp(200, _gemini_ok(json.dumps(GOOD)))])
    assert explainer.provider() == "gemini"
    out = asyncio.run(explainer.explain("Traffic piles onto a few routes", MATCH))
    assert out and out["steps"] == GOOD["steps"] and out["model"] == "gemini-test"
    assert calls == explainer.GEMINI_MODELS[:1]


def test_gemini_busy_falls_through_to_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _use_gemini(monkeypatch, [
        _gemini_resp(503, {"error": {"code": 503}}),
        _gemini_resp(200, _gemini_ok(json.dumps(GOOD))),
    ])
    out = asyncio.run(explainer.explain("Traffic", MATCH))
    assert out is not None and calls == explainer.GEMINI_MODELS[:2]


@pytest.mark.parametrize("responses", [
    [_gemini_resp(403, {"error": {"code": 403}})],
    [_gemini_resp(200, _gemini_ok("not json"))],
    [_gemini_resp(200, _gemini_ok(json.dumps(GOOD), finish="SAFETY")), _gemini_resp(200, _gemini_ok("", finish="SAFETY"))],
    [httpx.ConnectTimeout("t")] * 4,
    [_gemini_resp(429, {})] * 4,
])
def test_gemini_failures_return_none(monkeypatch: pytest.MonkeyPatch, responses) -> None:
    _use_gemini(monkeypatch, list(responses))
    assert asyncio.run(explainer.explain("Traffic", MATCH)) is None


def test_gemini_retries_once_when_all_busy_then_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _use_gemini(monkeypatch, [_gemini_resp(429, {}), _gemini_resp(429, {}),
                                      _gemini_resp(200, _gemini_ok(json.dumps(GOOD)))])
    out = asyncio.run(explainer.explain("Traffic", MATCH))
    assert out is not None and len(calls) == 3
    again = asyncio.run(explainer.explain("Traffic", MATCH))  # served from cache
    assert again == out and len(calls) == 3


def test_gemini_invalid_json_falls_through_to_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _use_gemini(monkeypatch, [_gemini_resp(200, _gemini_ok('{"headline": "trunc')),
                                      _gemini_resp(200, _gemini_ok(json.dumps(GOOD)))])
    assert asyncio.run(explainer.explain("Traffic", MATCH)) is not None and len(calls) == 2


def test_jargon_is_translated_in_output(monkeypatch: pytest.MonkeyPatch) -> None:
    jargon = dict(GOOD, insight="Use verification gates on 2-hop neighbors to collapse R_eff below 1.0.")
    _use_gemini(monkeypatch, [_gemini_resp(200, _gemini_ok(json.dumps(jargon)))])
    out = asyncio.run(explainer.explain("Misinformation spreads", MATCH))
    assert out is not None
    low = out["insight"].lower()
    assert "verification gate" not in low and "r_eff" not in low and "2-hop" not in low
    assert "automatic safety checks" in low and "stop the spread" in low


def test_precomputed_card_lookup() -> None:
    explainer._plain = {"bio-ant-pheromone": {"insight": "x", "steps": [], "how_it_works": "", "caveat": ""}}
    assert explainer.precomputed("bio-ant-pheromone")["model"] == "pre-generated"
    assert explainer.precomputed("missing") is None
    explainer._plain = None
