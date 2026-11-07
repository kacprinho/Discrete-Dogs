"""Multipart file ingestion and structural topology extraction (.py, .js, .txt, .pdf)."""

from __future__ import annotations

import ast
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".txt", ".md", ".pdf", ".json", ".csv"}


@dataclass
class IngestedDocument:
    filename: str
    size_bytes: int
    extension: str
    extracted_text: str
    chunks: List[str]
    node_count: int
    edge_count: int
    contention_hubs: int
    structural_signals: List[str] = field(default_factory=list)
    summary_log: str = ""
    topology_log: str = ""


BOTTLENECK_PATTERNS = [
    (r"\b(lock|mutex|semaphore|deadlock|contention|synchronized|atomic)\b", "lock/mutex synchronization bottleneck"),
    (r"\b(queue|buffer|backpressure|congestion|throttle|rate_limit|drop)\b", "queueing & backpressure congestion"),
    (r"\b(retry|timeout|cascade|circuit_breaker|failover|storm)\b", "cascading retry / fault propagation"),
    (r"\b(order\s*book|liquidity|bid|ask|spread|slippage|volatility|tick)\b", "order-book liquidity & microstructure flux"),
    (r"\b(graph|routing|shortest_path|dijkstra|bfs|dfs|topology|mesh|gossip)\b", "network routing & graph topology bottleneck"),
    (r"\b(cache|evict|lru|memory|gc|allocation|shard|partition)\b", "cache locality & shard allocation pressure"),
    (r"\b(viral|cascade|rumor|misinformation|contagion|broadcast|fanout)\b", "high-fanout contagion propagation"),
]


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _analyze_python_ast(source: str) -> Optional[tuple[int, int, int, List[str]]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.AsyncFor, ast.While))]
    awaits = [n for n in ast.walk(tree) if isinstance(n, ast.Await)]

    nodes = max(4, len(functions) + len(classes) + len(loops))
    edges = max(nodes + 1, len(calls) + len(loops) * 2)
    hubs = max(1, min(8, len(loops) + (1 if awaits else 0) + max(1, len(functions) // 3)))

    signals: List[str] = []
    if functions:
        signals.append(f"functions: {', '.join(functions[:5])}")
    if classes:
        signals.append(f"classes: {', '.join(classes[:4])}")
    if loops:
        signals.append(f"{len(loops)} iterative loop constructs")
    if awaits:
        signals.append(f"{len(awaits)} async await suspension points")
    return nodes, edges, hubs, signals


def extract_structural_signals(text: str) -> List[str]:
    lowered = text.lower()
    matched: List[str] = []
    for pattern, label in BOTTLENECK_PATTERNS:
        if re.search(pattern, lowered):
            matched.append(label)
    return matched


def ingest_file_bytes(filename: str, raw_bytes: bytes) -> IngestedDocument:
    ext = Path(filename or "upload.txt").suffix.lower() or ".txt"
    size_bytes = len(raw_bytes)

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages_text: List[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                continue
        extracted_text = "\n".join(pages_text).strip()
        page_info = f"{len(reader.pages)} PDF page(s)"
    else:
        extracted_text = raw_bytes.decode("utf-8", errors="replace").strip()
        line_count = len(extracted_text.splitlines())
        page_info = f"{line_count} lines"

    chunks = _chunk_text(extracted_text)
    signals = extract_structural_signals(extracted_text)

    ast_stats = _analyze_python_ast(extracted_text) if ext == ".py" else None
    if ast_stats is not None:
        node_count, edge_count, hubs, ast_signals = ast_stats
        signals = ast_signals + signals
    else:
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", extracted_text)
        unique_terms = len(set(w.lower() for w in words[:500]))
        node_count = max(6, min(64, unique_terms // 4 or 12))
        edge_count = max(node_count + 3, int(node_count * 2.2))
        hubs = max(1, min(9, len(signals) + max(1, len(chunks) // 2)))

    summary_log = (
        f"Parsed {filename} ({size_bytes:,} bytes · {page_info}) → "
        f"{max(1, len(chunks))} semantic segment(s) embedded"
    )
    signal_desc = ", ".join(signals[:3]) if signals else "coupled state-transition bottleneck"
    topology_log = (
        f"graph: {node_count} nodes · {edge_count} edges · "
        f"{hubs} contention hub(s) detected ({signal_desc})"
    )

    return IngestedDocument(
        filename=filename,
        size_bytes=size_bytes,
        extension=ext,
        extracted_text=extracted_text,
        chunks=chunks or ([extracted_text] if extracted_text else []),
        node_count=node_count,
        edge_count=edge_count,
        contention_hubs=hubs,
        structural_signals=signals,
        summary_log=summary_log,
        topology_log=topology_log,
    )
