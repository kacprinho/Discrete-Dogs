import { forwardRef, useImperativeHandle, useState } from "react";
import type { TeleportResult, Vec3 } from "@/lib/teleport-api";

// INTEGRATION SLOT for codex's Three.js canvas.
// Keep this props + ref interface when swapping in the WebGL implementation:
//   props.result  — latest TeleportResult (null before first run); draw the bezier bridge
//                   source_point → target_point and highlight the matched node.
//   props.busy    — true while the stream is running (e.g. pulse/spin the clusters).
//   props.label   — "Source → Target" caption.
//   ref.teleportCamera(coords) — smoothly lerp the camera to coords.
// The console calls teleportCamera(result.target_point) as soon as a `result` event arrives.

export type GalaxyCanvasHandle = { teleportCamera: (target: Vec3) => void };

type Props = { result: TeleportResult | null; busy: boolean; label: string };

// Placeholder renderer: projects the two points onto an SVG so the flow is visible pre-WebGL.
export const GalaxyCanvas = forwardRef<GalaxyCanvasHandle, Props>(function GalaxyCanvas(
  { result, busy, label },
  ref,
) {
  const [focus, setFocus] = useState<Vec3 | null>(null);
  useImperativeHandle(ref, () => ({ teleportCamera: (t) => setFocus(t) }), []);

  const px = (v: Vec3) => ({ x: 50 + v[0] * 4, y: 50 - v[1] * 4 });
  const s = result ? px(result.source_point) : null;
  const t = result ? px(result.target_point) : null;

  return (
    <div className="absolute inset-0">
      {result && s && t ? (
        <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" className="h-full w-full">
          <path
            d={`M ${s.x} ${s.y} Q 50 ${Math.min(s.y, t.y) - 25} ${t.x} ${t.y}`}
            fill="none"
            stroke="var(--color-accent, gold)"
            strokeWidth="0.6"
            strokeDasharray="2 1"
          />
          <circle cx={s.x} cy={s.y} r="2.5" fill="cyan" opacity="0.9" />
          <circle cx={t.x} cy={t.y} r={focus ? 3.5 : 2.5} fill="gold" opacity="0.9" />
          <text x={t.x} y={t.y + 7} fontSize="2.6" fill="currentColor" textAnchor="middle">
            {result.match.title.slice(0, 48)}
          </text>
        </svg>
      ) : (
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <div
              className={`orb mx-auto mb-4 h-24 w-24 rounded-full ${busy ? "animate-pulse" : ""}`}
            />
            <p className="text-xs tracking-[0.4em] text-muted-foreground">THREE.JS VIEWPORT</p>
            <p className="mt-1 text-sm text-accent">{label}</p>
          </div>
        </div>
      )}
    </div>
  );
});

export default GalaxyCanvas;
