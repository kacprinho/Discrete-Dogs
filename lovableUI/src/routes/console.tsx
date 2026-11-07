import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Paperclip, X } from "lucide-react";
import GalaxyCanvas, { type GalaxyCanvasHandle } from "@/components/GalaxyCanvas";
import { streamTeleport, type TeleportEvent, type TeleportResult } from "@/lib/teleport-api";

export const Route = createFileRoute("/console")({
  head: () => ({
    meta: [
      { title: "Latent Teleportation — Agent Console" },
      {
        name: "description",
        content:
          "Cross-domain problem translation dashboard with galaxy viewport and agent reasoning terminal.",
      },
      { property: "og:title", content: "Latent Teleportation — Agent Console" },
      {
        property: "og:description",
        content: "Translate bottlenecks across domains via latent vector offsets.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Console,
});

const PRESETS: { label: string; domain: string }[] = [
  { label: "Traffic -> Ants", domain: "natural" },
  { label: "Order Book -> Fluid Dynamics", domain: "physics" },
  { label: "Database Lock -> Enzymes", domain: "natural" },
  { label: "Misinformation -> Virus", domain: "social" },
];

const DOMAINS = [
  { id: "natural", label: "🌿 Natural & Biological Systems (e.g., Ant Pheromones, Mycelium)" },
  { id: "physics", label: "🌊 Applied Physics & Fluid Dynamics (e.g., Navier-Stokes, Turbulence)" },
  {
    id: "social",
    label: "🦠 Social Systems & Epidemiology (e.g., Viral Spread, Ring Inoculation)",
  },
  { id: "math", label: "🔢 Pure & Applied Mathematics (e.g., Graph Topology, Entropy)" },
  { id: "cognitive", label: "🧠 Cognitive Science & Neuroscience (e.g., Synaptic Pruning)" },
  { id: "universal", label: "🌐 Universal Cross-Domain Search (Auto-Detect)" },
];

const ACCEPT = ".py,.js,.pdf,.txt";

type Line = { stage: string; text: string; kind: "stage" | "info" | "user" | "done" | "error" };

function formatSize(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(2)} MB`;
}

function Console() {
  const [active, setActive] = useState<string>(PRESETS[0]!.label);
  const [lines, setLines] = useState<Line[]>([
    { stage: "SYS", text: "agent kernel booted · awaiting bottleneck", kind: "info" },
  ]);
  const [file, setFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [domain, setDomain] = useState("universal");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TeleportResult | null>(null);
  const canvasRef = useRef<GalaxyCanvasHandle>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [lines]);

  const pickFile = (f: File | undefined) => {
    if (!f) return;
    const ext = "." + (f.name.split(".").pop() ?? "").toLowerCase();
    if (!ACCEPT.split(",").includes(ext)) {
      setLines((p) => [
        ...p,
        { stage: "ERROR", text: `unsupported file type: ${f.name}`, kind: "error" },
      ]);
      return;
    }
    setFile(f);
  };

  const push = (l: Line) => setLines((p) => [...p, l]);

  const onEvent = (ev: TeleportEvent) => {
    switch (ev.type) {
      case "log": {
        const kind: Line["kind"] =
          ev.stage === "DONE" ? "done" : /—/.test(ev.text) ? "stage" : "info";
        push({ stage: ev.stage, text: ev.text, kind });
        break;
      }
      case "result":
        setResult(ev);
        canvasRef.current?.teleportCamera(ev.target_point, ev.source_point, {
          targetTitle: ev.match.title,
          domain: ev.match.domain,
          score: ev.match.score,
          candidates: ev.candidates,
        });
        push({
          stage: "MATCH",
          text: `${ev.match.title} · ${ev.match.domain} · sim ${ev.match.score.toFixed(2)}`,
          kind: "done",
        });
        break;
      case "error":
        push({ stage: "ERROR", text: ev.text, kind: "error" });
        break;
      case "done":
        break;
    }
  };

  const teleport = async (overrides?: { description?: string; domain?: string }) => {
    const desc = overrides?.description ?? description;
    if (busy || (!file && !desc.trim())) return;
    const domainId = overrides?.domain ?? domain;
    const domainLabel = DOMAINS.find((d) => d.id === domainId)?.label ?? domainId;
    setBusy(true);
    setResult(null);
    canvasRef.current?.reset();
    push({
      stage: "INPUT",
      text: `${file ? `file: ${file.name} · ` : ""}context: "${desc.trim() || "—"}" → ${domainLabel}`,
      kind: "user",
    });

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamTeleport(
        { file, description: desc, domain: domainId, domainLabel, signal: ctrl.signal },
        onEvent,
      );
    } catch (e) {
      if (!ctrl.signal.aborted)
        push({ stage: "ERROR", text: `teleport failed: ${(e as Error).message}`, kind: "error" });
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  };

  const kindClass = (k: Line["kind"]) =>
    k === "stage"
      ? "text-primary"
      : k === "user"
        ? "text-accent"
        : k === "done"
          ? "text-glow text-primary"
          : k === "error"
            ? "text-destructive"
            : "text-foreground/80";

  return (
    <div className="cyber-grid flex h-screen flex-col bg-background font-mono text-foreground lg:flex-row">
      {/* LEFT 60% */}
      <section className="flex min-h-[50vh] flex-col gap-3 border-border p-4 lg:w-3/5 lg:border-r">
        <div className="flex items-center justify-between gap-3">
          <Link
            to="/"
            className="group flex min-w-0 items-center gap-2"
            aria-label="Back to landing screen"
          >
            <span className="text-sm text-muted-foreground transition-colors group-hover:text-primary">
              ←
            </span>
            <h1 className="text-glow truncate text-lg font-bold tracking-[0.3em] text-primary">
              LATENT//TELEPORTATION
            </h1>
          </Link>
          <span className="shrink-0 text-xs text-muted-foreground">domain preset</span>
        </div>
        <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              disabled={busy}
              onClick={() => {
                setActive(p.label);
                setDomain(p.domain);
                const [src = p.label, tgt = ""] = p.label.split("->");
                teleport({
                  description: `${src.trim()} bottleneck — find analogue in ${tgt.trim()}`,
                  domain: p.domain,
                });
              }}
              data-active={active === p.label}
              className="preset-btn rounded-md border px-3 py-2 text-xs uppercase tracking-wider disabled:opacity-40"
            >
              {p.label.replace("->", "→")}
            </button>
          ))}
        </div>
        <div
          id="galaxy-viewport"
          className="viewport relative flex-1 overflow-hidden rounded-lg border border-border"
        >
          <div className="absolute inset-0">
            <GalaxyCanvas
              ref={canvasRef}
              sourceLabel={active.split("->")[0]?.trim() || "Your problem"}
              targetLabel={
                result ? `${result.match.domain} · ${result.match.title}` : "Target domain"
              }
              result={result}
            />
          </div>
        </div>
      </section>

      {/* RIGHT 40% */}
      <section className="flex min-h-[50vh] flex-col lg:w-2/5">
        <header className="flex flex-wrap gap-4 border-b border-border px-4 py-3 text-xs">
          <Status label="FAISS Vector Index" value="Online" />
          <Status label="Models" value="Dual Gemini / Claude / Codex" tone="accent" />
        </header>
        <div
          ref={scrollRef}
          className="terminal min-h-40 flex-1 space-y-1 overflow-y-auto p-4 text-xs leading-relaxed"
        >
          {lines.map((l, i) => (
            <div key={i} className="animate-in fade-in slide-in-from-bottom-1">
              <span className="text-muted-foreground">[{l.stage.padEnd(9, " ")}]</span>{" "}
              <span className={kindClass(l.kind)}>{l.text}</span>
            </div>
          ))}
          {busy && <div className="caret text-primary">▋</div>}
        </div>

        {result && <MatchCard result={result} />}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            setActive("Your problem");
            teleport();
          }}
          className="space-y-3 border-t border-border p-3 text-xs"
        >
          {/* Dropzone */}
          {file ? (
            <div className="flex items-center gap-2 rounded-md border border-primary/50 bg-card px-3 py-2">
              <Paperclip className="h-4 w-4 text-primary" />
              <span className="flex-1 truncate text-foreground">{file.name}</span>
              <span className="text-muted-foreground">{formatSize(file.size)}</span>
              <button
                type="button"
                aria-label="Remove file"
                onClick={() => setFile(null)}
                className="rounded p-1 text-muted-foreground hover:text-destructive"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                pickFile(e.dataTransfer.files[0]);
              }}
              data-active={dragging}
              className="preset-btn flex cursor-pointer flex-col items-center gap-1 rounded-md border border-dashed px-3 py-4 text-center"
            >
              <Paperclip className="h-5 w-5" />
              <span>Drop a file or click to upload</span>
              <span className="text-[10px] text-muted-foreground">.py · .js · .pdf · .txt</span>
            </div>
          )}
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPT}
            hidden
            onChange={(e) => {
              pickFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />

          <label className="block space-y-1">
            <span className="uppercase tracking-wider text-muted-foreground">
              Problem / File Context
            </span>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              placeholder="describe the bottleneck or what the file contains…"
              className="w-full resize-none rounded-md border border-border bg-card px-3 py-2 text-foreground outline-none placeholder:text-muted-foreground focus:border-primary"
            />
          </label>

          <label className="block space-y-1">
            <span className="uppercase tracking-wider text-muted-foreground">
              Target Knowledge Domain
            </span>
            <select
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              className="w-full rounded-md border border-border bg-card px-3 py-2 text-foreground outline-none focus:border-primary"
            >
              {DOMAINS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>

          <button
            type="submit"
            disabled={busy || (!file && !description.trim())}
            data-active="true"
            className="preset-btn w-full rounded-md border py-2.5 text-sm uppercase tracking-[0.3em] disabled:opacity-40"
          >
            {busy ? "Teleporting…" : "Teleport"}
          </button>
        </form>
      </section>
    </div>
  );
}

function Status({
  label,
  value,
  tone = "primary",
}: {
  label: string;
  value: string;
  tone?: "primary" | "accent";
}) {
  return (
    <div className="flex items-center gap-2">
      <span className={`status-dot ${tone === "accent" ? "bg-accent" : "bg-primary"}`} />
      <span className="text-muted-foreground">{label}:</span>
      <span className={tone === "accent" ? "text-accent" : "text-primary"}>{value}</span>
    </div>
  );
}

function MatchCard({ result }: { result: TeleportResult }) {
  const m = result.match;
  return (
    <div className="max-h-56 space-y-1 overflow-y-auto border-t border-border bg-card/60 p-3 text-xs">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-bold text-primary">
          {m.url ? (
            <a href={m.url} target="_blank" rel="noreferrer" className="underline">
              {m.title}
            </a>
          ) : (
            m.title
          )}
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {m.source === "arxiv-live" && (
            <span className="rounded border border-accent px-1 text-[10px] uppercase tracking-wider text-accent">
              live · arXiv
            </span>
          )}
          <span className="text-accent">sim {m.score.toFixed(2)}</span>
        </span>
      </div>
      <p className="text-muted-foreground">
        {m.domain}
        {m.mechanism ? ` · mechanism: ${m.mechanism}` : ""}
      </p>
      {m.paper_ref && <p className="text-muted-foreground/80 italic">{m.paper_ref}</p>}
      <p className="text-foreground/80">{m.summary}</p>
      {m.adaptation?.length ? (
        <ul className="list-inside list-disc text-foreground/80">
          {m.adaptation.map((a, i) => (
            <li key={i}>{a}</li>
          ))}
        </ul>
      ) : null}
      {m.limitations?.length ? (
        <p className="text-destructive/80">limits: {m.limitations.join("; ")}</p>
      ) : null}
      {result.candidates?.length ? (
        <p className="text-muted-foreground">
          also:{" "}
          {result.candidates
            .slice(0, 3)
            .map((c) => c.title)
            .join(" · ")}
        </p>
      ) : null}
    </div>
  );
}
