import { createFileRoute } from "@tanstack/react-router";

// POST /api/teleport — see src/lib/teleport-api.ts for the wire contract.
//
// If TELEPORT_BACKEND_URL is set (e.g. http://localhost:8000), the multipart request is proxied
// to the FastAPI engine at `${TELEPORT_BACKEND_URL}/api/teleport` and its NDJSON stream is piped
// straight through. Otherwise an error event is returned, unless TELEPORT_MOCK=1 opts into clearly
// labelled demo data (never presented as retrieval evidence).

function env(name: string): string | undefined {
  try {
    return typeof process !== "undefined" ? process.env?.[name] : undefined;
  } catch {
    return undefined;
  }
}

function backendUrl(): string | undefined {
  const v = env("TELEPORT_BACKEND_URL");
  return v ? v.replace(/\/+$/, "") : undefined;
}

const NDJSON = {
  "Content-Type": "application/x-ndjson; charset=utf-8",
  "Cache-Control": "no-cache",
};

async function proxy(base: string, form: FormData): Promise<Response> {
  try {
    const upstream = await fetch(`${base}/api/teleport`, { method: "POST", body: form });
    if (!upstream.ok || !upstream.body) {
      const detail = (await upstream.text().catch(() => "")).slice(0, 300);
      return ndjson([
        { type: "error", text: `backend HTTP ${upstream.status}${detail ? ` — ${detail}` : ""}` },
        { type: "done" },
      ]);
    }
    return new Response(upstream.body, { headers: NDJSON });
  } catch (e) {
    return ndjson([
      { type: "error", text: `backend unreachable at ${base}: ${(e as Error).message}` },
      { type: "done" },
    ]);
  }
}

function ndjson(events: unknown[], delayMs = 0): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      for (const ev of events) {
        controller.enqueue(enc.encode(JSON.stringify(ev) + "\n"));
        if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      }
      controller.close();
    },
  });
  return new Response(stream, { headers: NDJSON });
}

function mock(form: FormData): Response {
  const file = form.get("file");
  const description = String(form.get("description") ?? "").slice(0, 2000);
  const domain = String(form.get("domain") ?? "universal").slice(0, 100);
  const label = String(form.get("domain_label") ?? domain).slice(0, 200);
  const hasFile = file instanceof File;
  const log = (stage: string, agent: string, text: string) => ({ type: "log", stage, agent, text });

  const events: unknown[] = [];
  if (hasFile) {
    events.push(
      log("INGEST", "ingest", `Ingesting file vectors... ${file.name} (${file.size} bytes)`),
    );
    events.push(
      log(
        "INGEST",
        "ingest",
        `chunked into ${Math.max(1, Math.ceil(file.size / 2048))} segments · embedded`,
      ),
    );
  }
  events.push(
    log(
      "TOPOLOGY",
      "topology",
      `Topology Extraction — ${description ? `"${description.slice(0, 80)}"` : "inferring from file"}`,
    ),
    log("TOPOLOGY", "topology", "graph: 14 nodes · 31 edges · 3 contention hubs detected"),
    log("LATENT", "latent", "Latent Vector Offset — v = v_prob − v_source + v_target · FAISS k=8"),
    log("TRANSLATE", "translate", `Domain Translation — projecting into ${label}`),
    {
      type: "result",
      source_point: [-6, 1.5, 0],
      target_point: [6, -1, 2],
      offset_norm: 0.4182,
      match: {
        id: "mock-ant-pheromone",
        title: "Ant colony pheromone trail reinforcement (MOCK)",
        domain: "natural",
        summary:
          "Ants balance load across paths via evaporating pheromone trails: busy routes are reinforced, stale ones decay.",
        score: 0.87,
        mechanism: "positive feedback with decay (stigmergy)",
        adaptation: ["Weight routes by recent success", "Decay weights over time to avoid lock-in"],
        limitations: ["Mock result — backend not connected (set TELEPORT_BACKEND_URL)"],
      },
    },
    log("DONE", "system", "candidate solution emitted (mock backend)"),
    { type: "done" },
  );
  return ndjson(events, 350);
}

export const Route = createFileRoute("/api/teleport")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const form = await request.formData();
        const base = backendUrl();
        if (base) return proxy(base, form);
        if (env("TELEPORT_MOCK") === "1") return mock(form);
        return ndjson([
          {
            type: "error",
            text: "backend not configured — set TELEPORT_BACKEND_URL (e.g. http://localhost:8000), or TELEPORT_MOCK=1 for demo data",
          },
          { type: "done" },
        ]);
      },
    },
  },
});
