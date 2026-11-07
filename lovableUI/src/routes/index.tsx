import { createFileRoute, Link } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Latent Teleportation — Translate problems across domains" },
      {
        name: "description",
        content: "Teleport bottlenecks into analogous domains through latent vector offsets.",
      },
      { property: "og:title", content: "Latent Teleportation" },
      {
        property: "og:description",
        content: "Teleport bottlenecks into analogous domains through latent vector offsets.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Landing,
});

function Landing() {
  return (
    <main className="cyber-grid viewport relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-background px-6 font-mono text-foreground">
      <div className="orb absolute h-72 w-72 rounded-full opacity-40" />
      <div className="relative text-center">
        <p className="mb-4 text-xs tracking-[0.5em] text-muted-foreground">// AGENT KERNEL v0.1</p>
        <h1 className="text-glow text-4xl font-bold tracking-[0.25em] text-primary md:text-6xl">
          LATENT
          <br />
          TELEPORTATION
        </h1>
        <p className="mx-auto mt-6 max-w-md text-sm text-muted-foreground">
          Translate any bottleneck into a domain that has already solved it.
        </p>
        <Link
          to="/console"
          data-active="true"
          className="preset-btn mt-10 inline-block rounded-md border px-8 py-3 text-sm uppercase tracking-[0.3em]"
        >
          Get Started →
        </Link>
      </div>
    </main>
  );
}
