// Client + shared types for POST /api/teleport.
//
// WIRE CONTRACT (v1) — agreed with backend (FastAPI) and canvas (GalaxyCanvas):
//
// Request: multipart/form-data
//   file?         File  (.py | .js | .txt | .pdf)
//   description   string  — problem / file context (may be empty if file given)
//   domain        string  — target domain id: natural | physics | social | math | cognitive | universal
//   domain_label  string  — human-readable label (for logs only)
//
// Response: 200, Content-Type: application/x-ndjson — one JSON object per line:
//   {"type":"log",    "stage":"INGEST", "agent":"ingest", "text":"..."}
//   {"type":"result", "source_point":[x,y,z], "target_point":[x,y,z],
//                     "match":{...Match}, "candidates":[...Match], "offset_norm":0.41}
//   {"type":"error",  "text":"..."}
//   {"type":"done"}
// Coordinates are 3D projections (PCA/UMAP) of the 384-d MiniLM vectors, roughly in [-10, 10].
//
// Legacy plain-text lines "STAGE|text" are still accepted for back-compat with the Lovable mock.

export type Vec3 = [number, number, number];

export type Match = {
  id?: string;
  title: string;
  domain: string;
  summary: string;
  score: number; // cosine similarity, 0..1
  url?: string;
  mechanism?: string; // shared structural mechanism, e.g. "positive-feedback reinforcement"
  adaptation?: string[]; // concrete steps to apply it to the source problem
  limitations?: string[];
  paper_ref?: string; // citation, e.g. "Smith et al., arXiv:1502.05855 (2015)"
  source?: "arxiv-live" | string; // set when the match came from the live arXiv fallback
  point?: Vec3;
  coords?: Vec3;
  source_cosine?: number;
  offset_cosine?: number;
  relevance_score?: number;
  passes_source_cosine_gate?: boolean;
  passes_offset_cosine_gate?: boolean;
  below_threshold?: boolean;
  situation_description?: string;
  /** Optional plain-English explanation from the LLM layer (absent if no key / failure). */
  explanation?: {
    headline: string;
    connection: string;
    steps: string[];
    caveats: string;
    model?: string;
  };
  why_it_maps?: string;
};

export type TeleportResult = {
  source_point: Vec3;
  target_point: Vec3;
  match: Match;
  candidates?: Match[];
  offset_norm?: number;
  source_cosine?: number;
  offset_cosine?: number;
  relevance_score?: number;
  passes_source_cosine_gate?: boolean;
  passes_offset_cosine_gate?: boolean;
  corpus_size?: number;
  below_threshold?: boolean;
};

export type TeleportEvent =
  | { type: "log"; stage: string; agent?: string; text: string }
  | ({ type: "result" } & TeleportResult)
  | { type: "error"; text: string }
  | { type: "done" };

export type TeleportRequest = {
  file: File | null;
  description: string;
  domain: string;
  domainLabel: string;
  signal?: AbortSignal;
};

export function parseLine(raw: string): TeleportEvent | null {
  const line = raw.trim();
  if (!line) return null;
  if (line.startsWith("{")) {
    try {
      const ev = JSON.parse(line) as TeleportEvent;
      if (ev && typeof ev === "object" && "type" in ev) return ev;
    } catch {
      /* fall through to legacy */
    }
  }
  const [stage = "INFO", ...rest] = line.split("|");
  if (stage === "DONE") return { type: "log", stage, text: rest.join("|") };
  return { type: "log", stage, text: rest.length ? rest.join("|") : line };
}

/** Streams events from /api/teleport, invoking onEvent for each parsed line. */
export async function streamTeleport(req: TeleportRequest, onEvent: (ev: TeleportEvent) => void) {
  const body = new FormData();
  if (req.file) body.append("file", req.file);
  body.append("description", req.description);
  body.append("domain", req.domain);
  body.append("domain_label", req.domainLabel);

  const res = await fetch("/api/teleport", { method: "POST", body, signal: req.signal ?? null });
  if (!res.ok || !res.body) {
    let detail = "";
    try {
      detail = (await res.text()).slice(0, 200);
    } catch {
      /* ignore */
    }
    throw new Error(`HTTP ${res.status}${detail ? ` — ${detail}` : ""}`);
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n");
    buf = parts.pop() ?? "";
    for (const p of parts) {
      const ev = parseLine(p);
      if (ev) onEvent(ev);
    }
  }
  const tail = parseLine(buf + dec.decode());
  if (tail) onEvent(tail);
}
