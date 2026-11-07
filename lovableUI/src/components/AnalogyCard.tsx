import type { Match, TeleportResult } from "@/lib/teleport-api";

// Executive-level result card. The main body is plain English only; anything academic
// (paper titles, mechanisms, scores, the vector formula, citations) lives in "Behind the Math".

const FIELD_NAMES: Record<string, string> = {
  natural: "nature & biology",
  physics: "physics",
  social: "epidemiology & social science",
  math: "mathematics",
  cognitive: "neuroscience",
};

function fieldName(m: Match): string {
  const id = (m as Match & { domain_id?: string }).domain_id;
  if (id && FIELD_NAMES[id]) return FIELD_NAMES[id]!;
  return m.domain.replace(/\s*&\s*/g, " & ").toLowerCase();
}

function strengthLabel(score: number): { text: string; tone: string } {
  if (score >= 0.55) return { text: "Strong match", tone: "border-primary text-primary" };
  if (score >= 0.4) return { text: "Good match", tone: "border-accent text-accent" };
  return { text: "Exploratory match", tone: "border-muted-foreground text-muted-foreground" };
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
      {children}
    </p>
  );
}

export function AnalogyCard({ result, problem }: { result: TeleportResult; problem: string }) {
  const m = result.match;
  const b = m.explanation;
  const s = strengthLabel(m.score);
  const live = m.source === "arxiv-live";
  const fallbackSteps = (m.adaptation ?? []).map((a) => ({ title: a, detail: "" }));
  const steps = b?.steps?.length ? b.steps : fallbackSteps;

  return (
    <div className="max-h-[50vh] shrink-0 space-y-3 overflow-y-auto border-t border-border bg-card/60 p-3 text-xs leading-relaxed">
      {/* Core insight */}
      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className={`rounded border px-1.5 text-[10px] uppercase tracking-wider ${s.tone}`}>
            {s.text}
          </span>
          <span className="rounded border border-border px-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
            Borrowed from {fieldName(m)}
          </span>
          {live && (
            <span className="rounded border border-accent px-1.5 text-[10px] uppercase tracking-wider text-accent">
              Fresh research
            </span>
          )}
        </div>
        <p className="text-sm font-bold leading-snug text-primary">
          {b?.insight ?? `A proven idea from ${fieldName(m)} that fits this problem.`}
        </p>
        {problem && <p className="text-[11px] text-muted-foreground">For: {problem}</p>}
      </div>

      {/* What to do */}
      {steps.length > 0 && (
        <div className="space-y-1">
          <Label>What to do</Label>
          <ol className="space-y-1">
            {steps.slice(0, 3).map((st, i) => (
              <li key={i} className="flex gap-2">
                <span className="mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-primary text-[10px] text-primary">
                  {i + 1}
                </span>
                <span>
                  <span className="font-bold text-foreground">{st.title}</span>
                  {st.detail && <span className="text-foreground/80"> — {st.detail}</span>}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* How it works */}
      {!b && (
        <div className="space-y-1">
          <Label>How it works</Label>
          <p className="text-foreground/90">{m.summary}</p>
          <p className="text-[10px] text-muted-foreground">
            Add a Gemini key to the backend for a plain-English action plan on fresh research
            results.
          </p>
        </div>
      )}

      {(b?.how_it_works || b?.caveat) && (
        <div className="space-y-1">
          <Label>How it works</Label>
          {b?.how_it_works && <p className="text-foreground/90">{b.how_it_works}</p>}
          {b?.caveat && (
            <p className="text-foreground/70">
              <span className="font-semibold">Watch out:</span> {b.caveat}
            </p>
          )}
        </div>
      )}

      {/* Behind the Math */}
      <details className="group rounded border border-border/70 bg-background/40 px-2 py-1.5">
        <summary className="cursor-pointer select-none text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground group-open:text-primary">
          Behind the Math
        </summary>
        <div className="mt-2 space-y-2 text-[11px] text-foreground/80">
          <div>
            <p className="font-semibold text-foreground">Research source</p>
            <p>
              {m.url ? (
                <a href={m.url} target="_blank" rel="noreferrer" className="text-primary underline">
                  {m.title}
                </a>
              ) : (
                m.title
              )}
              {m.paper_ref ? ` — ${m.paper_ref}` : ""}
            </p>
            {m.mechanism && <p className="italic">Mechanism: {m.mechanism}</p>}
          </div>
          <div>
            <p className="font-semibold text-foreground">Latent teleportation</p>
            <p className="font-mono">
              v_target = normalize(v_problem − α·v_source_field + β·v_target_field)
            </p>
            <p>
              cosine(problem, paper) = {(m.source_cosine ?? m.score).toFixed(3)}
              {m.offset_cosine !== undefined && (
                <> · cosine(v_target, paper) = {m.offset_cosine.toFixed(3)}</>
              )}
              {result.offset_norm !== undefined && <> · ‖Δv‖ = {result.offset_norm.toFixed(3)}</>}
            </p>
            <p className="text-muted-foreground">
              all-MiniLM-L6-v2 embeddings (384-d) · FAISS inner-product search
              {result.corpus_size ? ` over ${result.corpus_size} mechanisms` : ""}
              {live ? " · retrieved live from arXiv" : ""}
            </p>
          </div>
          {m.adaptation?.length ? (
            <div>
              <p className="font-semibold text-foreground">Original technical adaptation</p>
              <ul className="list-inside list-disc">
                {m.adaptation.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {m.limitations?.length ? (
            <p>
              <span className="font-semibold text-foreground">Known limitations:</span>{" "}
              {m.limitations.join(" ")}
            </p>
          ) : null}
          {result.candidates?.length ? (
            <p>
              <span className="font-semibold text-foreground">Other analogies:</span>{" "}
              {result.candidates
                .slice(0, 3)
                .map((c) => `${c.title} (${c.score.toFixed(2)})`)
                .join(" · ")}
            </p>
          ) : null}
          {b?.model && (
            <p className="text-[10px] text-muted-foreground">
              Plain-English briefing{" "}
              {b.model === "pre-generated" ? "pre-generated" : `written by ${b.model}`} from this
              source only.
            </p>
          )}
        </div>
      </details>
    </div>
  );
}

export default AnalogyCard;
