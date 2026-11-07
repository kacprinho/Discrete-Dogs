import { createFileRoute } from "@tanstack/react-router";

// Mock teleport endpoint: accepts multipart form (file, description, domain) and streams reasoning lines.
export const Route = createFileRoute("/api/teleport")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const form = await request.formData();
        const file = form.get("file");
        const description = String(form.get("description") ?? "").slice(0, 2000);
        const domain = String(form.get("domain") ?? "universal").slice(0, 100);
        const hasFile = file instanceof File;

        const lines: string[] = [];
        if (hasFile) {
          lines.push(`INGEST|Ingesting file vectors... ${file.name} (${file.size} bytes)`);
          lines.push(`INGEST|chunked into ${Math.max(1, Math.ceil(file.size / 2048))} segments · embedded`);
        }
        lines.push(`TOPOLOGY|Topology Extraction — ${description ? `"${description.slice(0, 80)}"` : "inferring from file"}`);
        lines.push("TOPOLOGY|graph: 14 nodes · 31 edges · 3 contention hubs detected");
        lines.push("LATENT|Latent Vector Offset — FAISS k=8 nearest analogues · Δ‖v‖ = 0.4182");
        lines.push(`TRANSLATE|Domain Translation — projecting into ${domain}`);
        lines.push("TRANSLATE|mapped 3 mechanisms · confidence 0.87 (Gemini/Claude consensus)");
        lines.push("DONE|candidate solution emitted → Codex synthesis queued");

        const enc = new TextEncoder();
        const stream = new ReadableStream({
          async start(controller) {
            for (const l of lines) {
              controller.enqueue(enc.encode(l + "\n"));
              await new Promise((r) => setTimeout(r, 400));
            }
            controller.close();
          },
        });
        return new Response(stream, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
      },
    },
  },
});
