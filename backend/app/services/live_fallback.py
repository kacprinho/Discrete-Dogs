"""Live research fallback: when local FAISS confidence is low, pull papers from arXiv,
embed them on the fly, inject them into the runtime index, and return the best one.

Trigger rule uses RAW cosine(v_prob, best local doc), not the calibrated display score:
calibrated = 0.55 + 0.42*cos is floored at 0.52, so it cannot separate good from bad matches.
Measured on the 22-doc corpus: presets have raw 0.43-0.64, off-corpus free text 0.05-0.18.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .arxiv_client import ArxivClient, ArxivPaper, build_queries

logger = logging.getLogger("teleport.fallback")

MIN_RAW_COS = float(os.environ.get("FALLBACK_MIN_RAW_COS", "0.30"))
FALLBACK_TIMEOUT = float(os.environ.get("FALLBACK_TIMEOUT_S", "15"))
FETCH_PDF = os.environ.get("FALLBACK_FETCH_PDF", "1") == "1"
ENABLED = os.environ.get("FALLBACK_ENABLED", "1") == "1"

_client = ArxivClient(timeout=float(os.environ.get("ARXIV_HTTP_TIMEOUT_S", "8")))
_index_lock = threading.Lock()


@dataclass
class FallbackOutcome:
    used: bool
    logs: List[Dict[str, Any]] = field(default_factory=list)
    match: Optional[Dict[str, Any]] = None
    target_point: Optional[List[float]] = None
    raw_similarity: Optional[float] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)


def should_fallback(raw_similarity: float, force: bool = False) -> bool:
    return ENABLED and (force or raw_similarity < MIN_RAW_COS)


def _log(stage: str, text: str) -> Dict[str, Any]:
    return {"type": "log", "stage": stage, "agent": "arxiv", "text": text}


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _teleport_vector(engine: Any, v_prob: np.ndarray, domain: str) -> np.ndarray:
    """Same offset the engine uses; prefers an engine-provided method if one exists."""
    if hasattr(engine, "compute_target_vector"):
        return engine.compute_target_vector(v_prob, domain)
    _, v_source = engine._infer_source_domain(v_prob)
    v_dom = engine.target_centroids.get(domain)
    if v_dom is None:
        return v_prob
    return _l2(v_prob - 0.35 * v_source + 0.55 * v_dom)


def _best_passage(engine: Any, text: str, v_prob: np.ndarray, max_sentences: int = 120) -> str:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if 60 <= len(s.strip()) <= 400 and not re.search(r"arxiv|©|copyright|doi\.org", s, re.I)
    ][:max_sentences]
    if not sentences:
        return ""
    sims = engine.encode(sentences) @ v_prob
    return sentences[int(np.argmax(sims))]


def _inject(engine: Any, doc: Dict[str, Any], vec: np.ndarray) -> None:
    """Appends a live paper to the runtime corpus + FAISS index (idempotent by id)."""
    with _index_lock:
        if any(d["id"] == doc["id"] for d in engine.corpus):
            return
        engine.corpus.append(doc)
        engine.corpus_embeddings = np.vstack([engine.corpus_embeddings, vec.reshape(1, -1)]).astype(
            np.float32
        )
        engine.faiss_index.add(vec.reshape(1, -1).astype(np.float32))


def _to_doc(p: ArxivPaper, domain: str, domain_label: str, mechanism: str) -> Dict[str, Any]:
    authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
    return {
        "id": f"arxiv:{p.arxiv_id}",
        "title": p.title,
        "domain": domain,
        "domain_label": domain_label,
        "summary": p.abstract[:600] + ("…" if len(p.abstract) > 600 else ""),
        "abstract": p.abstract,
        "url": p.url,
        "paper_ref": f"{authors}, arXiv:{p.arxiv_id} ({p.published[:4]})",
        "mechanism": mechanism or None,
        "adaptation": [],
        "limitations": [
            "Retrieved live from arXiv and ranked by embedding similarity; the analogy is not human-curated.",
        ],
        "source": "arxiv-live",
    }


async def run_fallback(
    engine: Any,
    problem_text: str,
    domain: str,
    domain_label: str,
    local_raw: float,
    force: bool = False,
) -> FallbackOutcome:
    try:
        return await asyncio.wait_for(
            _run(engine, problem_text, domain, domain_label, local_raw, force), FALLBACK_TIMEOUT
        )
    except asyncio.TimeoutError:
        return FallbackOutcome(
            used=False,
            logs=[_log("ARXIV", f"arXiv fallback timed out after {FALLBACK_TIMEOUT:.0f}s — keeping local match")],
        )
    except Exception as exc:
        logger.warning("fallback failed: %s", exc)
        return FallbackOutcome(
            used=False,
            logs=[_log("ARXIV", f"arXiv unreachable ({type(exc).__name__}) — keeping local match")],
        )


async def _run(
    engine: Any, problem_text: str, domain: str, domain_label: str, local_raw: float, force: bool
) -> FallbackOutcome:
    out = FallbackOutcome(used=False)
    if force:
        out.logs.append(_log("ARXIV", "Live search requested — querying arXiv"))
    else:
        out.logs.append(
            _log("ARXIV", f"Low-confidence local match (raw_cos={local_raw:.3f} < {MIN_RAW_COS:.2f}) — querying arXiv live")
        )

    queries = build_queries(problem_text, domain)
    if not queries:
        out.logs.append(_log("ARXIV", "No usable keywords in the problem text — keeping local match"))
        return out

    papers: Dict[str, ArxivPaper] = {}
    for q in queries:
        found = await _client.search(q, max_results=10)
        out.logs.append(_log("ARXIV", f"query {q[:110]} → {len(found)} papers"))
        for p in found:
            if p.abstract and p.arxiv_id not in papers:
                papers[p.arxiv_id] = p
        if len(papers) >= 8:
            break
    if not papers:
        out.logs.append(_log("ARXIV", "arXiv returned no papers — keeping local match"))
        return out

    plist = list(papers.values())
    v_prob = engine.encode([problem_text])[0]
    v_target = _teleport_vector(engine, v_prob, domain)
    vecs = await asyncio.to_thread(engine.encode, [f"{p.title}. {p.abstract}" for p in plist])
    raw = vecs @ v_prob
    tele = vecs @ v_target
    # Rank by the teleported vector, but require the paper to stay related to the actual problem.
    score = 0.5 * tele + 0.5 * raw
    order = np.argsort(-score)
    out.logs.append(
        _log(
            "EMBED",
            f"Embedded {len(plist)} abstracts on the fly · best raw_cos={float(raw[order[0]]):.3f} "
            f"(local was {local_raw:.3f})",
        )
    )

    best_i = int(order[0])
    best = plist[best_i]
    mechanism = ""
    if FETCH_PDF and best.pdf_url and float(raw[best_i]) > local_raw:
        pdf_text = await _client.fetch_pdf_text(best)
        if pdf_text:
            mechanism = await asyncio.to_thread(_best_passage, engine, pdf_text, v_prob)
            out.logs.append(
                _log("PDF", f"Parsed PDF arXiv:{best.arxiv_id} with pypdf ({len(pdf_text):,} chars) · extracted key passage")
            )
        else:
            out.logs.append(_log("PDF", f"PDF for arXiv:{best.arxiv_id} unavailable — using abstract"))
    if not mechanism:
        mechanism = _best_passage(engine, best.abstract, v_prob)

    # Inject the top few into the runtime index so repeat queries hit them locally.
    injected: List[Dict[str, Any]] = []
    for i in order[:5]:
        p = plist[int(i)]
        doc = _to_doc(p, domain, domain_label, mechanism if p is best else "")
        _inject(engine, doc, vecs[int(i)])
        src_c = round(float(raw[int(i)]), 4)
        off_c = round(float(tele[int(i)]), 4)
        calibrated = round(float(np.clip(0.55 + 0.42 * float(score[int(i)]), 0.0, 0.99)), 4)
        cutoff = float(getattr(engine, "similarity_threshold", 0.65))
        try:
            m = engine._format_match(doc, src_c, emb=vecs[int(i)])
        except TypeError:
            m = engine._format_match(doc, src_c)
        m["source_cosine"] = src_c
        m["offset_cosine"] = off_c
        m["relevance_score"] = calibrated
        m["raw_similarity"] = src_c
        m["teleported_cosine"] = off_c
        m["passes_source_cosine_gate"] = bool(src_c >= cutoff)
        m["passes_offset_cosine_gate"] = bool(off_c >= cutoff)
        m["below_threshold"] = bool(src_c < cutoff)
        m["source"] = "arxiv-live"
        injected.append(m)
    out.logs.append(
        _log("INDEX", f"Injected {len(injected)} live papers into runtime FAISS (n={engine.faiss_index.ntotal})")
    )

    if float(raw[best_i]) <= local_raw:
        out.logs.append(
            _log("ARXIV", f"Best live paper (raw_cos={float(raw[best_i]):.3f}) does not beat the local match — keeping local")
        )
        return out

    out.used = True
    out.match = injected[0]
    out.candidates = injected[1:4]
    out.raw_similarity = round(float(raw[best_i]), 4)
    out.target_point = engine._project_3d(vecs[best_i], manifold="target")
    return out
