"""Offline tests for the arXiv live fallback (network is mocked)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.engine import get_engine
from app.services import live_fallback
from app.services.arxiv_client import ArxivPaper, build_queries, parse_atom

ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1502.05855v1</id>
    <title>Stochastic Cascade
      Amplification of Fluctuations</title>
    <published>2015-02-20T00:00:00Z</published>
    <summary>We show how multiplicative cascades amplify small fluctuations into
      heavy-tailed swings in coupled inventory-like reservoirs.</summary>
    <author><name>A. Author</name></author>
    <link href="https://arxiv.org/pdf/1502.05855v1" rel="related" type="application/pdf" title="pdf"/>
    <category term="cond-mat.stat-mech"/>
  </entry>
</feed>"""


def _paper(i: int, abstract: str) -> ArxivPaper:
    return ArxivPaper(f"9999.{i:05d}v1", f"Paper {i}", abstract, ["X"], "2024-01-01",
                      f"https://arxiv.org/abs/9999.{i:05d}v1", None)


def test_parse_atom_and_queries() -> None:
    papers = parse_atom(ATOM)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "1502.05855v1"
    assert p.title == "Stochastic Cascade Amplification of Fluctuations"
    assert p.pdf_url and p.pdf_url.endswith("1502.05855v1")
    qs = build_queries("supply chain bullwhip effect amplifies inventory swings", "physics")
    assert qs and "cascade" in qs[0] and "cat:physics.flu-dyn" in qs[0]


def test_preset2_never_triggers_fallback() -> None:
    c = get_engine().teleport("Order Book bottleneck — find analogue in Fluid Dynamics", "physics")
    assert c.match["id"] == "phys-navier-stokes-orderbook"
    assert not live_fallback.should_fallback(c.raw_similarity)


def test_low_confidence_injects_live_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = get_engine()
    text = "supply chain bullwhip effect amplifies inventory swings"
    c = eng.teleport(text, "physics")
    assert live_fallback.should_fallback(c.raw_similarity)

    async def fake_search(query: str, max_results: int = 8):
        return [
            _paper(1, "Bullwhip amplification of inventory swings in supply chains modelled as a cascade."),
            _paper(2, "Galaxy rotation curves and dark matter halos."),
        ]

    monkeypatch.setattr(live_fallback._client, "search", fake_search)
    n_before = eng.faiss_index.ntotal
    out = asyncio.run(live_fallback.run_fallback(eng, text, "physics", "Physics", c.raw_similarity))
    assert out.used and out.match["id"] == "arxiv:9999.00001v1"
    assert out.match["source"] == "arxiv-live"
    assert out.raw_similarity > c.raw_similarity
    assert eng.faiss_index.ntotal == n_before + 2
    assert len(eng.corpus) == eng.faiss_index.ntotal == eng.corpus_embeddings.shape[0]
    # idempotent: same papers again are not re-added
    asyncio.run(live_fallback.run_fallback(eng, text, "physics", "Physics", c.raw_similarity))
    assert eng.faiss_index.ntotal == n_before + 2


def test_offline_keeps_local_match(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(query: str, max_results: int = 8):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(live_fallback._client, "search", boom)
    eng = get_engine()
    out = asyncio.run(live_fallback.run_fallback(eng, "warehouse robots colliding", "natural", "Nat", 0.1))
    assert not out.used and out.match is None
    assert any("unreachable" in l["text"] for l in out.logs)
