"""End-to-end verification suite for the Latent Teleportation FastAPI backend (compatible with both pytest and python -m)."""

from __future__ import annotations

import asyncio
import io
import json
import numpy as np
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

from app.engine import SIMILARITY_THRESHOLD, compute_unit_offset_vector, get_engine
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
    assert eng.similarity_threshold == SIMILARITY_THRESHOLD == 0.65

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
            f"(score={res.match['score']}, raw_cos={res.raw_similarity}, "
            f"teleported_cos={res.match.get('teleported_cosine')}, "
            f"below_threshold={res.below_threshold}, filtered_out={res.filtered_out_count}, "
            f"src={res.source_point}, tgt={res.target_point})"
        )
        assert res.match["id"] == expected_id, f"Expected {expected_id}, got {res.match['id']}"
        assert res.match["score"] >= SIMILARITY_THRESHOLD
        assert res.below_threshold is False
        assert "point" in res.match and len(res.match["point"]) == 3
        assert all(c["score"] >= SIMILARITY_THRESHOLD for c in res.candidates)
        assert all("point" in c and len(c["point"]) == 3 for c in res.candidates)
        assert -7.5 <= res.source_point[0] <= -2.5
        assert 2.5 <= res.target_point[0] <= 7.5
        assert abs(res.source_point[1]) <= 0.275
        assert abs(res.target_point[1]) <= 0.275

    # Low-relevance / out-of-distribution queries must flag below_threshold=True (< 0.65)
    for ood_text, ood_dom in [
        ("battery degradation in electric vehicle fleets reduces range over time", "physics"),
        ("medieval sourdough bread baking crust caramelization temperature", "math"),
    ]:
        ood = eng.teleport(ood_text, ood_dom)
        print(
            f"OOD Query [{ood_dom}] '{ood_text}' -> {ood.match['id']} "
            f"(score={ood.match['score']}, raw_cos={ood.raw_similarity}, below_threshold={ood.below_threshold})"
        )
        assert ood.below_threshold is True
        assert ood.match["score"] < SIMILARITY_THRESHOLD


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
