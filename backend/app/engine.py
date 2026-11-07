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

# Strict semantic similarity cutoff threshold & source-bottleneck grounding floor
SIMILARITY_THRESHOLD: float = 0.65
RAW_SOURCE_COSINE_FLOOR: float = 0.22

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


def compute_unit_offset_vector(
    v_prob: np.ndarray,
    v_source_domain: np.ndarray,
    v_target_domain: np.ndarray,
    alpha: float = 0.45,
    beta: float = 0.55,
) -> Tuple[np.ndarray, float]:
    """
    Computes the unit-norm-preserving Latent Teleportation vector:
      1. Orthogonalize & unit-normalize the problem residual against the source domain centroid:
         v_invariant = normalize(v_prob - alpha * <v_prob, v_source> * v_source)
      2. Unit-normalize the cross-manifold shift vector:
         delta_manifold = normalize(v_target_domain - 0.5 * v_source_domain)
      3. Combine on the unit sphere S^(d-1) without magnitude drift:
         v_target = normalize(v_invariant + beta * delta_manifold)
    Returns (v_target, chord_offset_norm).
    """
    u_prob = _l2_normalize(v_prob)
    u_src = _l2_normalize(v_source_domain)
    u_tgt = _l2_normalize(v_target_domain)

    # Project out source-domain component proportional to collinearity <u_prob, u_src>
    proj_coeff = float(np.dot(u_prob, u_src))
    v_invariant = _l2_normalize(u_prob - (alpha * max(0.25, proj_coeff)) * u_src)

    # Unit-normalized manifold shift direction
    delta_manifold = _l2_normalize(u_tgt - 0.50 * u_src)

    # Geodesic chord combination on unit sphere (zero magnitude drift: ||v_target||_2 == 1.0)
    v_target = _l2_normalize(v_invariant + beta * delta_manifold)
    chord_offset_norm = round(float(np.linalg.norm(v_target - u_prob)), 4)
    return v_target, chord_offset_norm


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
    below_threshold: bool = False
    similarity_threshold: float = SIMILARITY_THRESHOLD
    filtered_out_count: int = 0


class LatentTeleportationEngine:
    """Manages sentence-transformers embeddings, FAISS nearest-neighbor search, and 3D manifold projection."""

    def __init__(self, corpus_path: Path = CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        self.corpus: List[Dict[str, Any]] = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.similarity_threshold: float = SIMILARITY_THRESHOLD
        self.raw_cosine_floor: float = RAW_SOURCE_COSINE_FLOOR
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
            return _l2_normalize(np.asarray(vecs, dtype=np.float32))
        return _deterministic_fallback_embed(texts, EMBEDDING_DIM)

    def add_documents(self, new_docs: List[Dict[str, Any]]) -> List[int]:
        """
        Dynamically embeds and appends new research documents (e.g. from live arXiv fallback)
        into the runtime corpus and in-memory FAISS index.
        """
        added_indices: List[int] = []
        existing_ids = {d.get("id") for d in self.corpus}
        docs_to_add = [d for d in new_docs if d.get("id") not in existing_ids]
        if not docs_to_add:
            return added_indices

        texts = [
            f"{doc['title']}. {doc.get('summary', '')} {doc.get('abstract', '')} {doc.get('mechanism', '')}"
            for doc in docs_to_add
        ]
        vecs = self.encode(texts)
        start_idx = len(self.corpus)
        self.corpus.extend(docs_to_add)
        self.corpus_embeddings = np.vstack([self.corpus_embeddings, vecs])
        self.faiss_index.add(vecs)
        for offset in range(len(docs_to_add)):
            added_indices.append(start_idx + offset)
        return added_indices

    def _infer_source_domain(self, v_prob: np.ndarray) -> Tuple[str, np.ndarray]:
        best_key = "distributed_systems"
        best_sim = -1.0
        for key, centroid in self.source_centroids.items():
            sim = float(np.dot(v_prob, centroid))
            if sim > best_sim:
                best_sim = sim
                best_key = key
        return best_key, self.source_centroids[best_key]

    def compute_target_vector(self, v_prob: np.ndarray, domain: str) -> np.ndarray:
        """
        Computes the unit-normalized target vector v_target for a given problem embedding
        and target domain (used by both local FAISS search and live arXiv fallback).
        """
        _, v_source = self._infer_source_domain(v_prob)
        v_dom = self.target_centroids.get(domain)
        if v_dom is None:
            return _l2_normalize(v_prob)
        v_target, _ = compute_unit_offset_vector(v_prob, v_source, v_dom)
        return v_target

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

    def _calibrate_similarity(self, teleported_cos: float, raw_source_cos: float) -> float:
        """
        Computes a source-bottleneck-grounded similarity score in [0, 0.97].
        Tightly constrains the score by raw cosine similarity against the source bottleneck
        so loosely associated cross-domain analogies fall below SIMILARITY_THRESHOLD (0.65).
        """
        base = 0.55 + 0.42 * float(teleported_cos)
        if raw_source_cos < self.raw_cosine_floor:
            # Penalize candidates that lack structural grounding to the source bottleneck
            grounding_ratio = max(0.0, float(raw_source_cos) / self.raw_cosine_floor)
            base = base * (0.45 + 0.55 * (grounding_ratio ** 1.2))
        return float(np.clip(base, 0.05, 0.97))

    def _format_match(
        self,
        doc: Dict[str, Any],
        score: float,
        point_3d: Optional[List[float]] = None,
        emb: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        clipped = round(float(np.clip(score, 0.0, 0.99)), 4)
        if point_3d is None and emb is not None:
            point_3d = self._project_3d(emb, manifold="target")
        if point_3d is None:
            doc_idx = next((i for i, d in enumerate(self.corpus) if d.get("id") == doc.get("id")), None)
            if doc_idx is not None and doc_idx < len(self.corpus_embeddings):
                point_3d = self._project_3d(self.corpus_embeddings[doc_idx], manifold="target")
        formatted: Dict[str, Any] = {
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
        if point_3d is not None:
            formatted["point"] = point_3d
            formatted["coords"] = point_3d
        return formatted

    def build_situation_description(
        self,
        problem_text: str,
        source_domain_key: str,
        match: Dict[str, Any],
        signals: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Generates a situation-specific description and structural mapping narrative
        explicitly linking the user's input bottleneck to the retrieved scientific mechanism.
        """
        clean_prob = " ".join((problem_text or "").strip().split())
        snippet = clean_prob[:110] + ("…" if len(clean_prob) > 110 else "")
        src_label = source_domain_key.replace("_", " ")
        sig_phrase = (
            ", ".join(signals[:2])
            if signals
            else "coupled flow saturation and feedback amplification"
        )
        mech = match.get("mechanism") or "decentralized feedback regulation"
        first_step = (
            match["adaptation"][0]
            if isinstance(match.get("adaptation"), list) and match["adaptation"]
            else match.get("summary", "")
        )
        situation_desc = (
            f"In your {src_label} scenario (\"{snippet}\"), the bottleneck arises from {sig_phrase}. "
            f"{match['title']} ({match.get('domain', 'Target Domain')}) resolves the same structural "
            f"topology via {mech.lower()} — {first_step[0].lower() + first_step[1:] if first_step else ''}"
        ).strip()
        if not situation_desc.endswith("."):
            situation_desc += "."
        why_it_maps = f"{src_label.title()} ({sig_phrase}) ↔ {match['title']} ({mech})"
        return situation_desc, why_it_maps

    def teleport(
        self,
        problem_text: str,
        target_domain: str = "universal",
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> TeleportComputation:
        cutoff = self.similarity_threshold if similarity_threshold is None else float(similarity_threshold)
        normalized_domain = (target_domain or "universal").strip().lower()
        if normalized_domain not in DOMAIN_LABELS:
            for k in DOMAIN_LABELS:
                if k in normalized_domain:
                    normalized_domain = k
                    break
            else:
                normalized_domain = "universal"

        v_prob = self.encode([problem_text])[0]
        source_domain_key, v_source_domain = self._infer_source_domain(v_prob)

        # Determine target domain centroid for unit-norm vector offset arithmetic
        if normalized_domain != "universal" and normalized_domain in self.target_centroids:
            chosen_target_domain = normalized_domain
            v_target_domain = self.target_centroids[chosen_target_domain]
        else:
            best_domain = "natural"
            best_resonance = -1.0
            for dom_key, dom_centroid in self.target_centroids.items():
                candidate_v, _ = compute_unit_offset_vector(v_prob, v_source_domain, dom_centroid)
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

        v_target, offset_norm = compute_unit_offset_vector(
            v_prob, v_source_domain, v_target_domain
        )

        # Query the FAISS index with the unit-normalized teleported vector v_target
        search_k = len(self.corpus)
        scores, indices = self.faiss_index.search(v_target.reshape(1, -1), search_k)
        raw_scores_vec = (self.corpus_embeddings @ v_prob).tolist()

        all_domain_matches: List[Dict[str, Any]] = []
        target_cluster_vecs: List[np.ndarray] = []
        filtered_out_count = 0

        for sim_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_idx = int(idx)
            doc = self.corpus[doc_idx]
            if doc.get("domain") != chosen_target_domain:
                continue
            raw_cos = float(raw_scores_vec[doc_idx])
            calibrated_score = self._calibrate_similarity(float(sim_score), raw_cos)
            pt_3d = self._project_3d(self.corpus_embeddings[doc_idx], manifold="target")
            match_dict = self._format_match(doc, calibrated_score, point_3d=pt_3d)
            match_dict["raw_similarity"] = round(raw_cos, 4)
            match_dict["teleported_cosine"] = round(float(sim_score), 4)
            all_domain_matches.append(match_dict)
            target_cluster_vecs.append(self.corpus_embeddings[doc_idx])

        if not all_domain_matches:
            top_idx = int(indices[0][0])
            doc = self.corpus[top_idx]
            raw_cos = float(raw_scores_vec[top_idx])
            calibrated_score = self._calibrate_similarity(float(scores[0][0]), raw_cos)
            pt_3d = self._project_3d(self.corpus_embeddings[top_idx], manifold="target")
            match_dict = self._format_match(doc, calibrated_score, point_3d=pt_3d)
            match_dict["raw_similarity"] = round(raw_cos, 4)
            match_dict["teleported_cosine"] = round(float(scores[0][0]), 4)
            all_domain_matches.append(match_dict)
            target_cluster_vecs.append(self.corpus_embeddings[top_idx])

        # Enforce strict similarity cutoff threshold (SIMILARITY_THRESHOLD = 0.65)
        passing_matches = [m for m in all_domain_matches if m["score"] >= cutoff]
        filtered_out_count = len(all_domain_matches) - len(passing_matches)

        primary_match = all_domain_matches[0]
        primary_doc_idx = next(
            (i for i, d in enumerate(self.corpus) if d["id"] == primary_match["id"]), 0
        )
        raw_sim = round(float(raw_scores_vec[primary_doc_idx]), 4)
        teleported_sim = primary_match["score"]
        below_threshold = bool(teleported_sim < cutoff or raw_sim < self.raw_cosine_floor)
        primary_match["below_threshold"] = below_threshold
        sit_desc, why_maps = self.build_situation_description(
            problem_text, source_domain_key, primary_match
        )
        primary_match["situation_description"] = sit_desc
        primary_match["why_it_maps"] = why_maps

        # Only include secondary candidates that strictly pass the 0.65 threshold
        filtered_candidates = [
            m for m in passing_matches if m["id"] != primary_match["id"]
        ][: max(0, top_k - 1)]

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
            candidates=filtered_candidates,
            backend_mode=self.backend_mode,
            below_threshold=below_threshold,
            similarity_threshold=cutoff,
            filtered_out_count=filtered_out_count,
        )


_ENGINE_INSTANCE: Optional[LatentTeleportationEngine] = None


def get_engine() -> LatentTeleportationEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = LatentTeleportationEngine()
    return _ENGINE_INSTANCE
