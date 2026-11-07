"""End-to-end verification suite for the Latent Teleportation FastAPI backend (compatible with both pytest and python -m)."""

from __future__ import annotations

import asyncio
import io
import json
import numpy as np
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

from app.engine import CANDIDATE_MIN_COSINE, SIMILARITY_THRESHOLD, compute_unit_offset_vector, get_engine
from app.ingest import ingest_file_bytes
from app.main import health, list_corpus, teleport_endpoint


def make_request(accept: str = "application/x-ndjson") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/teleport",
        "headers": [(b"accept", accept.encode("utf-8"))],
        "query_string": b"",
    }
    return Request(scope)


def test_unit_norm_offset_no_magnitude_drift() -> None:
    rng = np.random.default_rng(42)
    for _ in range(10):
        v_prob = rng.standard_normal(384).astype(np.float32)
        v_src = rng.standard_normal(384).astype(np.float32)
        v_tgt = rng.standard_normal(384).astype(np.float32)
        v_out, chord_norm = compute_unit_offset_vector(v_prob, v_src, v_tgt)
        norm = float(np.linalg.norm(v_out))
        assert abs(norm - 1.0) < 1e-5, f"Magnitude drift detected: ||v_target||={norm}"
        assert 0.0 <= chord_norm <= 2.0


def test_engine_presets_and_threshold_filtering() -> None:
    eng = get_engine()
    print(
        f"Engine initialized: mode={eng.backend_mode}, dim={eng.dimension}, "
        f"vectors={eng.faiss_index.ntotal}, threshold={eng.similarity_threshold}"
    )
    assert eng.dimension == 384
    assert eng.faiss_index.ntotal >= 20
    assert eng.similarity_threshold == SIMILARITY_THRESHOLD == 0.40

    queries = [
        ("Traffic bottleneck — find analogue in Ants", "natural", "bio-ant-pheromone"),
        ("Order Book bottleneck — find analogue in Fluid Dynamics", "physics", "phys-navier-stokes-orderbook"),
        ("Database Lock bottleneck — find analogue in Enzymes", "natural", "bio-allosteric-enzymes"),
        ("Misinformation bottleneck — find analogue in Virus", "social", "soc-sir-ring-inoculation"),
        ("Misinformation bottleneck", "social", "soc-sir-ring-inoculation"),
        ("Misinformation bottleneck", "universal", "soc-sir-ring-inoculation"),
    ]

    for desc, dom, expected_id in queries:
        res = eng.teleport(desc, dom)
        print(
            f"Query [{dom}] '{desc}' -> {res.match['id']} "
            f"(source_cos={res.source_cosine}, offset_cos={res.offset_cosine}, "
            f"relevance_score={res.relevance_score}, passes_gate={res.passes_source_cosine_gate}, "
            f"below_threshold={res.below_threshold}, filtered_out={res.filtered_out_count}, "
            f"src={res.source_point}, tgt={res.target_point})"
        )
        assert res.match["id"] == expected_id, f"Expected {expected_id}, got {res.match['id']}"
        assert res.match["score"] == res.source_cosine == res.raw_similarity
        assert res.match["offset_cosine"] == res.offset_cosine == res.teleported_similarity
        assert res.match["relevance_score"] == res.relevance_score
        assert res.passes_source_cosine_gate == (res.source_cosine >= SIMILARITY_THRESHOLD)
        assert res.below_threshold == (res.source_cosine < SIMILARITY_THRESHOLD)
        assert "point" in res.match and len(res.match["point"]) == 3
        assert all(c["source_cosine"] >= CANDIDATE_MIN_COSINE for c in res.candidates)
        assert all("point" in c and len(c["point"]) == 3 for c in res.candidates)
        assert -7.5 <= res.source_point[0] <= -2.5
        assert 2.5 <= res.target_point[0] <= 7.5
        assert abs(res.source_point[1]) <= 0.275
        assert abs(res.target_point[1]) <= 0.275

    # Preset 2: raw source_cosine 0.5982 passes the 0.40 relevance gate (it failed the old 0.65 gate)
    preset2 = eng.teleport("Order Book bottleneck — find analogue in Fluid Dynamics", "physics")
    assert preset2.match["id"] == "phys-navier-stokes-orderbook"
    assert abs(preset2.source_cosine - 0.5982) < 1e-3
    assert preset2.passes_source_cosine_gate is True
    assert preset2.below_threshold is False
    assert len(preset2.candidates) >= 1
    assert "find analogue" not in preset2.match["situation_description"]

    # High-overlap query (> 0.65 actual source_cosine) must pass the 0.65 gate with below_threshold=False
    high_cos = eng.teleport(
        "High-frequency limit order book viscosity, bid-ask order flow density waves, and Navier-Stokes Reynolds turbulence",
        "physics",
    )
    print(
        f"High-Cosine Query [physics] -> {high_cos.match['id']} "
        f"(source_cos={high_cos.source_cosine}, offset_cos={high_cos.offset_cosine}, "
        f"relevance_score={high_cos.relevance_score}, passes_gate={high_cos.passes_source_cosine_gate}, "
        f"below_threshold={high_cos.below_threshold})"
    )
    assert high_cos.match["id"] == "phys-navier-stokes-orderbook"
    assert high_cos.source_cosine >= SIMILARITY_THRESHOLD
    assert high_cos.passes_source_cosine_gate is True
    assert high_cos.below_threshold is False

    # Low-relevance / out-of-distribution queries must flag below_threshold=True (< 0.65)
    for ood_text, ood_dom in [
        ("battery degradation in electric vehicle fleets reduces range over time", "physics"),
        ("medieval sourdough bread baking crust caramelization temperature", "math"),
    ]:
        ood = eng.teleport(ood_text, ood_dom)
        print(
            f"OOD Query [{ood_dom}] '{ood_text}' -> {ood.match['id']} "
            f"(source_cos={ood.source_cosine}, offset_cos={ood.offset_cosine}, "
            f"relevance_score={ood.relevance_score}, below_threshold={ood.below_threshold})"
        )
        assert ood.below_threshold is True
        assert ood.passes_source_cosine_gate is False
        assert ood.source_cosine < SIMILARITY_THRESHOLD


def test_ndjson_streaming_endpoint() -> None:
    async def _run() -> None:
        health_res = await health()
        assert health_res["status"] == "online"
        assert health_res["corpus_size"] >= 80
        corpus_res = await list_corpus(domain=None)
        assert corpus_res["total"] >= 80
        assert all("point" in item and "coords" in item for item in corpus_res["items"][:5])

        sample_py = b"""
import asyncio

lock = asyncio.Lock()
queue = []

async def handle_order_flow(order):
    async with lock:
        while len(queue) > 100:
            await asyncio.sleep(0.01)
        queue.append(order)
"""
        ingested = ingest_file_bytes("router.py", sample_py)
        assert ingested.extension == ".py"
        assert ingested.node_count >= 4

        upload = UploadFile(
            file=io.BytesIO(sample_py),
            filename="router.py",
            headers=Headers({"content-type": "text/x-python"}),
        )
        resp = await teleport_endpoint(
            request=make_request(),
            file=upload,
            description="Mutex lock contention under bursty order flow",
            domain="natural",
            domain_label="Natural & Biological Systems",
            format="ndjson",
        )
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        lines = [json.loads(line) for line in "".join(chunks).strip().splitlines()]
        types = [l["type"] for l in lines]
        print("Streamed NDJSON event types:", types)
        assert "log" in types
        assert "result" in types
        assert types[-1] == "done"
        result_ev = next(l for l in lines if l["type"] == "result")
        assert "source_point" in result_ev and "target_point" in result_ev
        assert "source_coords" in result_ev and "target_coords" in result_ev
        assert "match" in result_ev and "candidates" in result_ev
        assert result_ev["match"].get("situation_description")
        assert result_ev["match"].get("why_it_maps")
        assert result_ev["match"].get("coords") == result_ev["match"].get("point")
        assert result_ev.get("ingestion") is not None
        assert result_ev["ingestion"]["filename"] == "router.py"
        assert result_ev.get("embedded_segments", 0) >= 2

        # File-only upload (.js) with no description + multi-chunk .txt + .pdf fallback
        sample_js = (
            ("const padding = 'header_comment ';\n" * 60)
            + """
async function matchOrderBook(bidQueue, askQueue) {
  while (bidQueue.length > 0 && askQueue.length > 0) {
    await flushOrderBookLiquidity(bidQueue, askQueue);
  }
}
"""
        ).encode("utf-8")
        js_ingested = ingest_file_bytes("orderbook_matcher.js", sample_js)
        assert js_ingested.extension == ".js"
        assert len(js_ingested.chunks) >= 2
        assert any("order-book" in s or "functions:" in s for s in js_ingested.structural_signals)

        js_upload = UploadFile(
            file=io.BytesIO(sample_js),
            filename="orderbook_matcher.js",
            headers=Headers({"content-type": "application/javascript"}),
        )
        js_resp = await teleport_endpoint(
            request=make_request(),
            file=js_upload,
            description="",
            domain="physics",
            domain_label="Applied Physics & Fluid Dynamics",
            format="json",
        )
        js_payload = json.loads(js_resp.body.decode("utf-8"))
        assert js_payload["ingestion"]["filename"] == "orderbook_matcher.js"
        assert js_payload["ingestion"]["chunk_count"] >= 2
        assert js_payload["embedded_segments"] >= 2
        assert "source_cosine" in js_payload and "offset_cosine" in js_payload and "relevance_score" in js_payload

        # Malformed/synthetic PDF bytes must not crash ingestion
        pdf_ingested = ingest_file_bytes("paper.pdf", b"%PDF-1.4\nLock contention and mutex deadlock in database shards")
        assert pdf_ingested.extension == ".pdf"
        assert pdf_ingested.semantic_summary

    asyncio.run(_run())


async def run_tests() -> None:
    test_unit_norm_offset_no_magnitude_drift()
    test_engine_presets_and_threshold_filtering()
    test_ndjson_streaming_endpoint()
    print("All backend verification tests passed!")


if __name__ == "__main__":
    test_unit_norm_offset_no_magnitude_drift()
    test_engine_presets_and_threshold_filtering()
    test_ndjson_streaming_endpoint()
    print("All backend verification tests passed!")
