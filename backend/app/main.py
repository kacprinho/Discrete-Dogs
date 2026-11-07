"""FastAPI server for Latent Teleportation (/api/teleport)."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import DOMAIN_LABELS, get_engine
from .ingest import extract_structural_signals, ingest_file_bytes
from .services.live_fallback import run_fallback, should_fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm sentence-transformers + FAISS index at startup
    get_engine()
    yield


app = FastAPI(
    title="Latent Teleportation Engine",
    description="Cross-domain structural analogy discovery engine powered by sentence-transformers and FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health() -> Dict[str, Any]:
    engine = get_engine()
    domain_counts: Dict[str, int] = {}
    for d in engine.corpus:
        dom = d.get("domain", "universal")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    return {
        "status": "online",
        "engine": engine.backend_mode,
        "faiss_vectors": int(engine.faiss_index.ntotal),
        "corpus_size": len(engine.corpus),
        "domain_counts": domain_counts,
        "dimension": engine.dimension,
        "domains": DOMAIN_LABELS,
    }


@app.get("/api/corpus")
async def list_corpus(domain: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    engine = get_engine()
    enriched: List[Dict[str, Any]] = []
    domain_counts: Dict[str, int] = {}
    for i, d in enumerate(engine.corpus):
        dom = d.get("domain", "universal")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        if domain and domain != "universal" and dom != domain:
            continue
        item = dict(d)
        if i < len(engine.corpus_embeddings):
            pt = engine._project_3d(engine.corpus_embeddings[i], manifold="target")
            item["point"] = pt
            item["coords"] = pt
        enriched.append(item)
    return {
        "total": len(enriched),
        "corpus_size": len(engine.corpus),
        "domain_counts": domain_counts,
        "items": enriched,
    }


@app.post("/api/teleport")
async def teleport_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    description: Optional[str] = Form(default=None),
    domain: Optional[str] = Form(default="universal"),
    domain_label: Optional[str] = Form(default=None),
    live: Optional[str] = Form(default=None),
    live_q: Optional[str] = Query(default=None, alias="live", description="1 = force live arXiv search"),
    format: Optional[str] = Query(default=None, description="ndjson (default) | legacy | json"),
):
    # Support both multipart/form-data and application/json requests seamlessly
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type and description is None and file is None:
        try:
            body = await request.json()
            description = body.get("description", "")
            domain = body.get("domain", "universal")
            domain_label = body.get("domain_label")
            live = str(body.get("live", "")) or None
        except Exception:
            pass

    desc_text = (description or "").strip()
    target_domain = (domain or "universal").strip()
    label_text = (domain_label or DOMAIN_LABELS.get(target_domain, target_domain)).strip()

    raw_bytes: Optional[bytes] = None
    filename: Optional[str] = None
    if file is not None and file.filename:
        filename = file.filename
        raw_bytes = await file.read()

    if not desc_text and not raw_bytes:
        err_events = [
            {
                "type": "error",
                "text": "Provide either a bottleneck description or an uploaded file (.py, .js, .txt, .pdf).",
            },
            {"type": "done"},
        ]
        return StreamingResponse(
            (json.dumps(ev) + "\n" for ev in err_events),
            media_type="application/x-ndjson; charset=utf-8",
            status_code=200,
        )

    engine = get_engine()
    ingested = ingest_file_bytes(filename, raw_bytes) if (filename and raw_bytes is not None) else None

    combined_problem_parts: List[str] = []
    if desc_text:
        combined_problem_parts.append(desc_text)
    if ingested and ingested.extracted_text:
        combined_problem_parts.append(ingested.extracted_text[:4000])
    combined_problem = "\n\n".join(combined_problem_parts).strip()

    computation = engine.teleport(
        problem_text=combined_problem,
        target_domain=target_domain,
        top_k=5,
    )

    # Live research fallback: low raw cosine against the local corpus -> query arXiv,
    # embed on the fly, inject into the runtime FAISS index, and use the best paper.
    local_raw_similarity = computation.raw_similarity
    local_teleported_similarity = computation.teleported_similarity
    force_live = any(
        isinstance(v, str) and v.strip().lower() in ("1", "true", "yes", "on") for v in (live, live_q)
    )
    fallback_logs: List[Dict[str, Any]] = []
    if should_fallback(computation.raw_similarity, force=force_live):
        fb = await run_fallback(
            engine,
            combined_problem,
            computation.target_domain,
            computation.target_domain_label,
            computation.raw_similarity,
            force=force_live,
        )
        fallback_logs = fb.logs
        if fb.used and fb.match and fb.target_point:
            local_match = dict(computation.match, title=f"{computation.match['title']} (local corpus)")
            computation.candidates = (fb.candidates + [local_match])[:4]
            computation.match = fb.match
            computation.target_point = fb.target_point
            computation.raw_similarity = fb.raw_similarity or computation.raw_similarity
            computation.source_cosine = float(fb.match.get("source_cosine", computation.raw_similarity))
            computation.offset_cosine = float(fb.match.get("offset_cosine", computation.offset_cosine))
            computation.relevance_score = float(fb.match.get("relevance_score", computation.relevance_score))
            computation.teleported_similarity = computation.offset_cosine
            computation.passes_source_cosine_gate = bool(computation.source_cosine >= computation.similarity_threshold)
            computation.passes_offset_cosine_gate = bool(computation.offset_cosine >= computation.similarity_threshold)
            computation.below_threshold = bool(not computation.passes_source_cosine_gate)

    signals = extract_structural_signals(combined_problem)
    sit_desc, why_maps = engine.build_situation_description(
        combined_problem,
        computation.inferred_source_domain,
        computation.match,
        signals=signals,
    )
    computation.match["situation_description"] = sit_desc
    computation.match["why_it_maps"] = why_maps
    if computation.target_point:
        computation.match.setdefault("point", computation.target_point)
        computation.match.setdefault("coords", computation.target_point)

    # Build structured agent reasoning events
    events: List[Dict[str, Any]] = []

    events.append(
        {
            "type": "log",
            "stage": "SYS",
            "agent": "kernel",
            "text": (
                f"Engine online ({engine.backend_mode} · {engine.dimension}-d · "
                f"FAISS IndexFlatIP n={engine.faiss_index.ntotal})"
            ),
        }
    )

    if ingested is not None:
        events.append(
            {
                "type": "log",
                "stage": "INGEST",
                "agent": "ingest",
                "text": f"Ingesting file vectors — {ingested.summary_log}",
            }
        )
        events.append(
            {
                "type": "log",
                "stage": "TOPOLOGY",
                "agent": "topology",
                "text": f"Topology Extraction — {ingested.topology_log}",
            }
        )
    else:
        signals = extract_structural_signals(desc_text)
        signal_str = ", ".join(signals[:2]) if signals else "coupled resource/flow bottleneck"
        events.append(
            {
                "type": "log",
                "stage": "TOPOLOGY",
                "agent": "topology",
                "text": f'Topology Extraction — "{desc_text[:90]}" ({signal_str})',
            }
        )

    events.append(
        {
            "type": "log",
            "stage": "LATENT",
            "agent": "latent",
            "text": (
                f"Latent Vector Offset — v_target = v_prob - v_{computation.inferred_source_domain} "
                f"+ v_{computation.target_domain} (||Δv||_2 = {computation.offset_norm:.4f})"
            ),
        }
    )

    events.append(
        {
            "type": "log",
            "stage": "SEARCH",
            "agent": "faiss",
            "text": (
                f"FAISS k-NN Search — manifold={computation.target_domain_label} · "
                f"source_cos={local_raw_similarity:.4f} · offset_cos={local_teleported_similarity:.4f} · "
                f"relevance={computation.relevance_score:.4f} · "
                f"gate_0.65={'PASS' if computation.passes_source_cosine_gate else 'FAIL (<0.65)'}"
            ),
        }
    )

    events.extend(fallback_logs)

    events.append(
        {
            "type": "log",
            "stage": "TRANSLATE",
            "agent": "translate",
            "text": (
                f"Domain Translation — mapped to [{computation.match['title']}] "
                f"via mechanism: {computation.match.get('mechanism', 'structural isomorphism')}"
            ),
        }
    )

    result_event: Dict[str, Any] = {
        "type": "result",
        # Primary schema fields (see lovableUI/src/lib/teleport-api.ts):
        "source_point": computation.source_point,
        "target_point": computation.target_point,
        "offset_norm": computation.offset_norm,
        "match": computation.match,
        "candidates": computation.candidates,
        # Field aliases kept for backward compatibility:
        "source_coords": computation.source_point,
        "target_coords": computation.target_point,
        "source_cluster_preview": computation.source_cluster_preview,
        "target_cluster_preview": computation.target_cluster_preview,
        "inferred_source_domain": computation.inferred_source_domain,
        "target_domain": computation.target_domain,
        "source_cosine": computation.source_cosine,
        "offset_cosine": computation.offset_cosine,
        "relevance_score": computation.relevance_score,
        "passes_source_cosine_gate": computation.passes_source_cosine_gate,
        "passes_offset_cosine_gate": computation.passes_offset_cosine_gate,
        "raw_similarity": computation.raw_similarity,
        "below_threshold": computation.below_threshold,
        "similarity_threshold": computation.similarity_threshold,
        "filtered_out_count": computation.filtered_out_count,
        "corpus_size": int(engine.faiss_index.ntotal),
        "situation_description": computation.match.get("situation_description"),
        "why_it_maps": computation.match.get("why_it_maps"),
        "match_source": computation.match.get("source", "local") if isinstance(computation.match, dict) else "local",
    }
    events.append(result_event)

    events.append(
        {
            "type": "log",
            "stage": "DONE",
            "agent": "system",
            "text": (
                f"Teleportation complete — {computation.match['title']} "
                f"({computation.match.get('paper_ref', computation.match['domain'])})"
            ),
        }
    )
    events.append({"type": "done"})

    accept_header = (request.headers.get("accept") or "").lower()
    effective_format = (format or "").lower()
    if not effective_format:
        if "text/plain" in accept_header:
            effective_format = "legacy"
        else:
            effective_format = "ndjson"

    if effective_format == "json":
        return JSONResponse(
            {
                **result_event,
                "logs": [e for e in events if e.get("type") == "log"],
            }
        )

    if effective_format == "legacy":
        async def legacy_stream() -> AsyncGenerator[str, None]:
            for ev in events:
                if ev["type"] == "log":
                    yield f"{ev['stage']}|{ev['text']}\n"
                    await asyncio.sleep(0.08)
                elif ev["type"] == "result":
                    coords_payload = {
                        "source_coords": ev["source_coords"],
                        "target_coords": ev["target_coords"],
                        "source_cluster_preview": ev["source_cluster_preview"],
                        "target_cluster_preview": ev["target_cluster_preview"],
                        "match": ev["match"],
                        "candidates": ev["candidates"],
                    }
                    yield f"COORDS|{json.dumps(coords_payload)}\n"
                    await asyncio.sleep(0.08)

        return StreamingResponse(
            legacy_stream(),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    async def ndjson_stream() -> AsyncGenerator[str, None]:
        for ev in events:
            yield json.dumps(ev) + "\n"
            await asyncio.sleep(0.08)

    return StreamingResponse(
        ndjson_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )
