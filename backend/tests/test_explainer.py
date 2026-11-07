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
    "headline": "Ants spread traffic by letting busy trails fade.",
    "connection": "Both systems overload popular paths.",
    "steps": ["Decay route weights", "Penalise slow routes", "Probe idle routes"],
    "caveats": "Needs enough traffic to learn.",
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(explainer, "_client", SimpleNamespace(messages=fake))
    return fake


def test_success_returns_structured_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, ("end_turn", json.dumps(GOOD)))
    out = asyncio.run(explainer.explain("Traffic piles onto a few routes", MATCH))
    assert out and out["headline"] == GOOD["headline"] and len(out["steps"]) == 3
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
    monkeypatch.delenv("EXPLAINER_ENABLED", raising=False)
    assert explainer.enabled() is False
    assert asyncio.run(explainer.explain("x", MATCH)) is None
