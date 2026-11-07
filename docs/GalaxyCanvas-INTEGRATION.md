# GalaxyCanvas integration

Place GalaxyCanvas.jsx and GalaxyCanvas.d.ts in lovableUI/src/components/.
Install runtime dependency: npm install three

```tsx
import { useRef } from 'react';
import GalaxyCanvas, { type GalaxyCanvasHandle } from '@/components/GalaxyCanvas';

const galaxy = useRef<GalaxyCanvasHandle>(null);
// Render inside a parent with a defined height:
<GalaxyCanvas ref={galaxy} sourceLabel="Your problem" targetLabel="Retrieved domain" />
// Only after a genuine successful retrieval:
galaxy.current?.teleportCamera([5, 0, 0], [-5, 0, 0]);
// At the start of a new request:
galaxy.current?.reset();
```

teleportCamera returns false if not mounted or if coordinates are not finite triples within [-20,20]. Defaults source to [-5,0,0]. Fixed decorative cluster centers are [-5,0,0] and [5,0,0]. Use these centers if the backend has no documented shared 3D projection. The bridge can connect arbitrary bounded display coordinates; cluster particles are illustrative and do not represent individual corpus records. Raw 384-dimensional embeddings are not accepted. No automatic API assumptions or fake match is included.

The component uses a deterministic 4,400-point layout, three additive tube layers for the bridge, frame-rate-independent camera interpolation, ResizeObserver, capped pixel ratio, reduced-motion camera behavior, WebGL failure text, and cleanup on unmount. It creates WebGL only in an effect, for SSR compatibility. Type declaration supports the existing TypeScript consumer without changing its tsconfig.

Existing repo issue: src/routes/api/teleport.ts is a mock that emits fabricated chunk/node/confidence lines. Replace or proxy it to FastAPI; do not present those lines as retrieval evidence. Existing console reads pipe-delimited plain text, not typed vector events. Agree on a real result event before connecting the camera.

No repository push or console route edits are included in this handoff.
