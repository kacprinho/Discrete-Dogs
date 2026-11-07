"""End-to-end verification suite for the Latent Teleportation FastAPI backend (compatible with both pytest and python -m)."""

from __future__ import annotations

import asyncio
import io
import json
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

from app.engine import get_engine
from app.ingest import ingest_file_bytes
from app.main import health, teleport_endpoint


def make_request(accept: str = "application/x-ndjson") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/teleport",
        "headers": [(b"accept", accept.encode("utf-8"))],
        "query_string": b"",
    }
    return Request(scope)


def test_engine_presets_and_freeform() -> None:
    eng = get_engine()
    print(f"Engine initialized: mode={eng.backend_mode}, dim={eng.dimension}, vectors={eng.faiss_index.ntotal}")
    assert eng.dimension == 384
    assert eng.faiss_index.ntotal >= 20

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
            f"(score={res.match['score']}, offset_norm={res.offset_norm}, "
            f"src={res.source_point}, tgt={res.target_point})"
        )
        assert res.match["id"] == expected_id, f"Expected {expected_id}, got {res.match['id']}"
        # Ensure 3D coordinates sit inside the GalaxyCanvas.jsx particle discs (y in [-0.275, +0.275])
        assert -7.5 <= res.source_point[0] <= -2.5
        assert 2.5 <= res.target_point[0] <= 7.5
        assert abs(res.source_point[1]) <= 0.275
        assert abs(res.target_point[1]) <= 0.275


def test_ndjson_streaming_endpoint() -> None:
    async def _run() -> None:
        health_res = await health()
        assert health_res["status"] == "online"

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

    asyncio.run(_run())


async def run_tests() -> None:
    test_engine_presets_and_freeform()
    test_ndjson_streaming_endpoint()
    print("All backend verification tests passed!")


if __name__ == "__main__":
    test_engine_presets_and_freeform()
    test_ndjson_streaming_endpoint()
    print("All backend verification tests passed!")
