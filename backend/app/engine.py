"""Latent Teleportation Vector Engine: sentence-transformers (all-MiniLM-L6-v2) + FAISS IndexFlatIP."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger("teleport.engine")

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.json"
EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"

SOURCE_DOMAIN_ANCHORS: Dict[str, str] = {
    "distributed_systems": (
        "Distributed systems engineering, microservice RPC congestion, packet routing, "
        "thread pool contention, load balancing, queue backpressure, and tail latency."
    ),
    "database_concurrency": (
        "Relational and distributed database concurrency control, mutex lock contention, "
        "read-write transaction starvation, deadlock resolution, and MVCC shard indexing."
    ),
    "quant_finance": (
        "High-frequency limit order book market microstructure, bid-ask liquidity crunches, "
        "execution slippage, order flow toxicity, and flash crash volatility cascades."
    ),
    "network_security": (
        "High-fanout broadcast amplification, gossip protocol flooding, unverified packet "
        "relaying across scale-free communication graphs, and blast-radius fault isolation."
    ),
}

TARGET_DOMAIN_ANCHORS: Dict[str, str] = {
    "natural": (
        "Natural and biological systems: ant colony pheromone stigmergy, Physarum slime mold "
        "flux tube dilation, allosteric enzyme cooperativity, mycelial nutrient networks, "
        "and adaptive immune clonal selection."
    ),
    "physics": (
        "Applied physics and fluid dynamics: compressible Navier-Stokes viscosity, Kolmogorov "
        "turbulence energy cascades, spin-glass parallel tempering replica exchange, Kuramoto "
        "coupled oscillator synchronization, and Landauer information thermodynamics."
    ),
    "social": (
        "Social systems and epidemiology: SIR ring inoculation, misinformation and viral "
        "contagion containment, scale-free acquaintance immunization, Granovetter complex "
        "contagion thresholds, Schelling spatial homophily, and congestion externality pricing."
    ),
    "math": (
        "Pure and applied mathematics: Ollivier-Ricci discrete graph curvature surgery, "
        "Ramanujan expander spectral gap, entropic Sinkhorn optimal transport, and "
        "persistent homology topological data analysis."
    ),
    "cognitive": (
        "Cognitive science and neuroscience: Turrigiano homeostatic synaptic pruning, "
        "Friston hierarchical predictive coding residuals, thalamic reticular inhibitory "
        "attention gating, and hippocampal sharp-wave ripple replay."
    ),
}

DOMAIN_LABELS: Dict[str, str] = {
    "natural": "Natural & Biological Systems",
    "physics": "Applied Physics & Fluid Dynamics",
    "social": "Social Systems & Epidemiology",
    "math": "Pure & Applied Mathematics",
    "cognitive": "Cognitive Science & Neuroscience",
    "universal": "Universal Cross-Domain Search",
}


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / norms


def _deterministic_fallback_embed(texts: List[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic char/word n-gram projection into 384-d used only if model weights cannot load."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = [t for t in text.lower().replace("-", " ").split() if len(t) >= 2]
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            for j in range(0, 24, 3):
                idx = ((h[j] << 8) | h[j + 1]) % dim
                sign = 1.0 if (h[j + 2] & 1) == 0 else -1.0
                out[row_idx, idx] += sign
    return _l2_normalize(out)


@dataclass
class TeleportComputation:
    inferred_source_domain: str
    target_domain: str
    target_domain_label: str
    offset_norm: float
    raw_similarity: float
    teleported_similarity: float
    source_point: List[float]
    target_point: List[float]
    source_cluster_preview: List[List[float]]
    target_cluster_preview: List[List[float]]
    match: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    backend_mode: str


class LatentTeleportationEngine:
    """Manages sentence-transformers embeddings, FAISS nearest-neighbor search, and 3D manifold projection."""

    def __init__(self, corpus_path: Path = CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        self.corpus: List[Dict[str, Any]] = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.model: Any = None
        self.backend_mode: str = "sentence-transformers/all-MiniLM-L6-v2"
        self._load_model()

        # Build corpus embeddings & global FAISS inner-product (cosine) index
        corpus_texts = [
            f"{doc['title']}. {doc['summary']} {doc.get('abstract', '')} {doc.get('mechanism', '')}"
            for doc in self.corpus
        ]
        self.corpus_embeddings = self.encode(corpus_texts)
        self.dimension = int(self.corpus_embeddings.shape[1])

        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(self.corpus_embeddings)

        # Precompute source and target domain centroids in latent space
        source_keys = list(SOURCE_DOMAIN_ANCHORS.keys())
        source_vecs = self.encode([SOURCE_DOMAIN_ANCHORS[k] for k in source_keys])
        self.source_centroids: Dict[str, np.ndarray] = {
            k: source_vecs[i] for i, k in enumerate(source_keys)
        }

        target_keys = list(TARGET_DOMAIN_ANCHORS.keys())
        target_vecs = self.encode([TARGET_DOMAIN_ANCHORS[k] for k in target_keys])
        self.target_centroids: Dict[str, np.ndarray] = {
            k: target_vecs[i] for i, k in enumerate(target_keys)
        }
        # Enhance each target centroid with the mean of its corpus documents
        for domain_id in target_keys:
            idxs = [i for i, d in enumerate(self.corpus) if d.get("domain") == domain_id]
            if idxs:
                combined = 0.5 * self.target_centroids[domain_id] + 0.5 * np.mean(
                    self.corpus_embeddings[idxs], axis=0
                )
                self.target_centroids[domain_id] = _l2_normalize(combined)

        # Fit a deterministic 3D PCA basis over the corpus + domain centroids
        all_vecs = np.vstack([self.corpus_embeddings, source_vecs, target_vecs])
        self.pca_mean = np.mean(all_vecs, axis=0, keepdims=True)
        centered = all_vecs - self.pca_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.pca_components = vt[:3].astype(np.float32)  # (3, 384)

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MODEL_NAME)
            logger.info("Loaded SentenceTransformer(%s)", MODEL_NAME)
        except Exception as exc:
            logger.warning("Falling back to deterministic 384-d embedder: %s", exc)
            self.model = None
            self.backend_mode = "deterministic-384d-fallback"

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return _deterministic_fallback_embed(texts, EMBEDDING_DIM)

    def _infer_source_domain(self, v_prob: np.ndarray) -> Tuple[str, np.ndarray]:
        best_key = "distributed_systems"
        best_sim = -1.0
        for key, centroid in self.source_centroids.items():
            sim = float(np.dot(v_prob, centroid))
            if sim > best_sim:
                best_sim = sim
                best_key = key
        return best_key, self.source_centroids[best_key]

    def _project_3d(self, vec: np.ndarray, manifold: str) -> List[float]:
        """
        Projects a 384-d vector onto the top 3 principal components and anchors:
          - source manifold in cyan galaxy disc (SOURCE = [-5, 0, 0], radius <= 2.8, y in [-0.25, +0.25])
          - target manifold in gold galaxy disc (TARGET = [+5, 0, 0], radius <= 2.8, y in [-0.25, +0.25])
        so the glowing Bezier bridge terminates directly inside the 3D particle discs in GalaxyCanvas.jsx.
        """
        centered = (vec.reshape(1, -1) - self.pca_mean) @ self.pca_components.T
        raw_x, raw_y, raw_z = [float(v) for v in centered[0]]
        if manifold == "source":
            x = -5.0 + float(np.tanh(raw_x * 2.2)) * 1.8
        else:
            x = 5.0 + float(np.tanh(raw_x * 2.2)) * 1.8
        y = float(np.tanh(raw_y * 2.0)) * 0.24
        z = float(np.tanh(raw_z * 2.2)) * 1.8
        return [round(x, 3), round(y, 3), round(z, 3)]

    def _format_match(self, doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        clipped = round(float(np.clip(score, 0.0, 0.99)), 4)
        return {
            "id": doc["id"],
            "title": doc["title"],
            "domain": doc.get("domain_label", DOMAIN_LABELS.get(doc["domain"], doc["domain"])),
            "domain_id": doc["domain"],
            "summary": doc["summary"],
            "score": clipped,
            "similarity": clipped,
            "url": doc.get("url"),
            "paper_ref": doc.get("paper_ref"),
            "mechanism": doc.get("mechanism"),
            "adaptation": doc.get("adaptation", []),
            "limitations": doc.get("limitations", []),
        }

    def teleport(
        self,
        problem_text: str,
        target_domain: str = "universal",
        top_k: int = 5,
    ) -> TeleportComputation:
        normalized_domain = (target_domain or "universal").strip().lower()
        if normalized_domain not in DOMAIN_LABELS:
            # Map descriptive labels if passed instead of ID
            for k in DOMAIN_LABELS:
                if k in normalized_domain:
                    normalized_domain = k
                    break
            else:
                normalized_domain = "universal"

        v_prob = self.encode([problem_text])[0]
        source_domain_key, v_source_domain = self._infer_source_domain(v_prob)

        # Determine target domain centroid for vector offset arithmetic:
        # v_target = v_prob - v_source_domain + v_target_domain
        if normalized_domain != "universal" and normalized_domain in self.target_centroids:
            chosen_target_domain = normalized_domain
            v_target_domain = self.target_centroids[chosen_target_domain]
        else:
            # In universal mode, evaluate offset across all target manifolds and pick strongest resonance
            best_domain = "natural"
            best_resonance = -1.0
            for dom_key, dom_centroid in self.target_centroids.items():
                candidate_v = _l2_normalize(v_prob - 0.35 * v_source_domain + 0.55 * dom_centroid)
                dom_idxs = [i for i, d in enumerate(self.corpus) if d.get("domain") == dom_key]
                if dom_idxs:
                    sims = 0.45 * (self.corpus_embeddings[dom_idxs] @ candidate_v) + 0.55 * (
                        self.corpus_embeddings[dom_idxs] @ v_prob
                    )
                    max_sim = float(np.max(sims))
                    if max_sim > best_resonance:
                        best_resonance = max_sim
                        best_domain = dom_key
            chosen_target_domain = best_domain
            v_target_domain = self.target_centroids[chosen_target_domain]

        v_target_raw = v_prob - 0.35 * v_source_domain + 0.55 * v_target_domain
        offset_norm = round(float(np.linalg.norm(v_target_raw - v_prob)), 4)
        v_target = _l2_normalize(v_target_raw)

        # Query the FAISS index with the teleported vector v_target
        search_k = len(self.corpus)
        scores, indices = self.faiss_index.search(v_target.reshape(1, -1), search_k)
        raw_scores_vec = (self.corpus_embeddings @ v_prob).tolist()

        ranked_matches: List[Dict[str, Any]] = []
        target_cluster_vecs: List[np.ndarray] = []

        for sim_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.corpus[int(idx)]
            if doc.get("domain") != chosen_target_domain:
                continue
            # Rescale inner product on unit sphere into calibrated similarity [0.65, 0.96]
            calibrated_score = float(np.clip(0.55 + 0.42 * float(sim_score), 0.52, 0.97))
            ranked_matches.append(self._format_match(doc, calibrated_score))
            target_cluster_vecs.append(self.corpus_embeddings[int(idx)])
            if len(ranked_matches) >= top_k:
                break

        # Fallback if a custom domain had no filtered items
        if not ranked_matches:
            top_idx = int(indices[0][0])
            doc = self.corpus[top_idx]
            calibrated_score = float(np.clip(0.55 + 0.42 * float(scores[0][0]), 0.52, 0.97))
            ranked_matches.append(self._format_match(doc, calibrated_score))
            target_cluster_vecs.append(self.corpus_embeddings[top_idx])

        primary_match = ranked_matches[0]
        primary_doc_idx = next(
            (i for i, d in enumerate(self.corpus) if d["id"] == primary_match["id"]), 0
        )
        raw_sim = round(float(raw_scores_vec[primary_doc_idx]), 4)
        teleported_sim = primary_match["score"]

        source_point = self._project_3d(v_prob, manifold="source")
        target_point = self._project_3d(
            self.corpus_embeddings[primary_doc_idx], manifold="target"
        )

        source_cluster_preview = [
            self._project_3d(vec, manifold="source")
            for vec in list(self.source_centroids.values())[:4]
        ]
        target_cluster_preview = [
            self._project_3d(vec, manifold="target") for vec in target_cluster_vecs[:4]
        ]

        return TeleportComputation(
            inferred_source_domain=source_domain_key,
            target_domain=chosen_target_domain,
            target_domain_label=DOMAIN_LABELS.get(chosen_target_domain, chosen_target_domain),
            offset_norm=offset_norm,
            raw_similarity=raw_sim,
            teleported_similarity=teleported_sim,
            source_point=source_point,
            target_point=target_point,
            source_cluster_preview=source_cluster_preview,
            target_cluster_preview=target_cluster_preview,
            match=primary_match,
            candidates=ranked_matches[1:4],
            backend_mode=self.backend_mode,
        )


_ENGINE_INSTANCE: Optional[LatentTeleportationEngine] = None


def get_engine() -> LatentTeleportationEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = LatentTeleportationEngine()
    return _ENGINE_INSTANCE
