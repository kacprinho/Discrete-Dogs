# Latent Teleportation — FastAPI + FAISS Vector Engine (`/backend`)

Cross-domain analogy discovery backend powered by `sentence-transformers` (`all-MiniLM-L6-v2`, 384-d) and an in-memory `faiss-cpu` (`IndexFlatIP`) cosine similarity index.

## Quick Start

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To connect the TanStack Start frontend (`lovableUI/`):
```bash
export TELEPORT_BACKEND_URL="http://localhost:8000"
```

## Mathematical Formulation

Given an uploaded file (`.py`, `.js`, `.txt`, `.pdf`) and/or bottleneck description:
1. Embed the problem context into $\mathbb{R}^{384}$ via `all-MiniLM-L6-v2`: $v_{\text{prob}}$.
2. Infer the closest source domain manifold centroid $v_{\text{source\_domain}}$ (`distributed_systems`, `database_concurrency`, `quant_finance`, `network_security`).
3. Select the target domain manifold centroid $v_{\text{target\_domain}}$ (`natural`, `physics`, `social`, `math`, `cognitive`, or optimal cross-domain manifold for `universal`).
4. Apply the Latent Teleportation vector offset:
   $$v_{\text{target}} = \text{normalize}\!\left(v_{\text{prob}} - v_{\text{source\_domain}} + v_{\text{target\_domain}}\right)$$
5. Query the `faiss.IndexFlatIP(384)` index and project the 384-d vectors via deterministic PCA into bounded 3D coordinates (`[-8.5, 8.5]`, with cyan source cluster in $-X$ and gold target cluster in $+X$).

## Endpoints

### `GET /health` (or `GET /api/health`)
Returns engine readiness, active embedding model, FAISS vector count, and supported domains.

### `POST /api/teleport`
Accepts `multipart/form-data` (`file`, `description`, `domain`, `domain_label`) or `application/json`.
Streams `application/x-ndjson` by default (`?format=legacy` streams `STAGE|TEXT` + `COORDS|{...}`; `?format=json` returns a single JSON object).

## Plain-English explanations (optional LLM layer)

The result card can be rewritten in plain English by an LLM, grounded only on the retrieved
paper. It is off unless a key is in the environment; without one the card shows the curated text.

```bash
export GEMINI_API_KEY=...        # Google AI Studio key (preferred, free tier)
# or: export ANTHROPIC_API_KEY=...
uvicorn app.main:app --port 8000
```

Optional settings: `EXPLAINER_PROVIDER` (auto | gemini | anthropic), `EXPLAINER_GEMINI_MODELS`
(default `gemini-3.5-flash-lite,gemini-3.5-flash`, tried in order), `EXPLAINER_TIMEOUT_S` (12).
Never commit keys; keep them in your shell or a file outside the repo.
