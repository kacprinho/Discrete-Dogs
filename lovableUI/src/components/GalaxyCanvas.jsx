import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import * as THREE from 'three';

const SOURCE = [-5, 0, 0];
const TARGET = [5, 0, 0];
const valid = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) && v.every((n) => Math.abs(n) <= 20);

/** Coordinates are display-space triples, never raw embedding dimensions. */
const GalaxyCanvas = forwardRef(function GalaxyCanvas({ className = '', sourceLabel = 'Source domain', targetLabel = 'Target domain' }, ref) {
  const host = useRef(null);
  const engine = useRef(null);
  const [error, setError] = useState('');
  const [matched, setMatched] = useState(false);

  useImperativeHandle(ref, () => ({
    teleportCamera(targetCoords, sourceCoords = SOURCE) {
      if (!valid(targetCoords) || !valid(sourceCoords) || !engine.current) return false;
      engine.current.teleport(targetCoords, sourceCoords);
      setMatched(true);
      return true;
    },
    reset() { engine.current?.reset(); setMatched(false); },
  }), []);

  useEffect(() => {
    const container = host.current;
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' }); }
    catch { setError('WebGL unavailable. Source and target results remain available in the terminal.'); return; }
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 150);
    const look = new THREE.Vector3();
    const desiredLook = new THREE.Vector3();
    const desiredCamera = new THREE.Vector3(0, 6, 21);
    camera.position.copy(desiredCamera);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x050814, 1);
    renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;';
    renderer.domElement.setAttribute('aria-hidden', 'true');
    container.appendChild(renderer.domElement);
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const resources = [];
    const track = (r) => { resources.push(r); return r; };
    let seed = 92837;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const sprite = track(new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexColors: true,
      vertexShader: 'varying vec3 tint; void main(){tint=color; vec4 p=modelViewMatrix*vec4(position,1.0); gl_PointSize=clamp(38.0/-p.z,1.0,7.0); gl_Position=projectionMatrix*p;}',
      fragmentShader: 'varying vec3 tint; void main(){float d=length(gl_PointCoord-0.5)*2.0; float a=1.0-smoothstep(0.0,1.0,d); gl_FragColor=vec4(tint,a*a);}',
    }));
    for (const [center, hex] of [[SOURCE, 0x22d3ee], [TARGET, 0xfbbf24]]) {
      const count = 2200;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      const color = new THREE.Color(hex);
      for (let i = 0; i < count; i++) {
        const radius = Math.sqrt(random()) * 3.2;
        const angle = radius * 1.7 + (i % 3) * Math.PI * 2 / 3 + (random() - 0.5) * 0.8;
        positions.set([center[0] + Math.cos(angle) * radius, (random() - 0.5) * 0.55, Math.sin(angle) * radius], i * 3);
        const intensity = 0.35 + random() * 0.65;
        colors.set([color.r * intensity, color.g * intensity, color.b * intensity], i * 3);
      }
      const geometry = track(new THREE.BufferGeometry());
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      scene.add(new THREE.Points(geometry, sprite));
    }
    const bridge = new THREE.Group();
    bridge.visible = false;
    scene.add(bridge);
    let bridgeGeometry;
    const bridgeMaterials = [0.07, 0.22, 0.9].map((opacity) => track(new THREE.MeshBasicMaterial({ color: 0x8ef8ed, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending })));
    function clearBridge() {
      for (const child of [...bridge.children]) { child.geometry.dispose(); bridge.remove(child); }
      bridgeGeometry = null;
    }
    engine.current = {
      teleport(target, source) {
        clearBridge();
        const start = new THREE.Vector3(...source);
        const end = new THREE.Vector3(...target);
        const arc = Math.max(2, start.distanceTo(end) * 0.3);
        const curve = new THREE.CubicBezierCurve3(start, start.clone().add(new THREE.Vector3(0, arc, 0)), end.clone().add(new THREE.Vector3(0, arc, 0)), end);
        [0.12, 0.055, 0.016].forEach((radius, i) => {
          bridgeGeometry = new THREE.TubeGeometry(curve, 80, radius, 6, false);
          bridge.add(new THREE.Mesh(bridgeGeometry, bridgeMaterials[i]));
        });
        bridge.visible = true;
        desiredLook.copy(end);
        desiredCamera.copy(end).add(new THREE.Vector3(0, 4, 11));
      },
      reset() { bridge.visible = false; desiredLook.set(0, 0, 0); desiredCamera.set(0, 6, 21); },
    };
    function resize() {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    let frame = 0;
    let previous = performance.now();
    let disposed = false;
    function render(now) {
      if (disposed) return;
      const dt = Math.min((now - previous) / 1000, 0.1);
      previous = now;
      const alpha = motion.matches ? 1 : 1 - Math.exp(-4 * dt);
      camera.position.lerp(desiredCamera, alpha);
      look.lerp(desiredLook, alpha);
      camera.lookAt(look);
      if (!document.hidden) renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    }
    frame = requestAnimationFrame(render);
    const lost = (event) => { event.preventDefault(); cancelAnimationFrame(frame); setError('Graphics context lost. Reload to restore the visualization.'); };
    renderer.domElement.addEventListener('webglcontextlost', lost);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      engine.current = null;
      clearBridge();
      resources.forEach((r) => r.dispose());
      renderer.domElement.removeEventListener('webglcontextlost', lost);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <section className={className} aria-label="Cross-domain galaxy visualization" style={{ position: 'relative', width: '100%', height: '100%', minHeight: 320, overflow: 'hidden', background: '#050814' }}>
    <div ref={host} style={{ position: 'absolute', inset: 0 }} />
    <div style={{ position: 'absolute', top: 16, left: 16, right: 16, display: 'flex', justifyContent: 'space-between', gap: 16, pointerEvents: 'none', fontSize: 12 }}>
      <span style={{ color: '#22d3ee' }}>{sourceLabel}</span><span style={{ color: '#fbbf24' }}>{targetLabel}</span>
    </div>
    <p aria-live="polite" style={{ position: 'absolute', bottom: 12, left: 16, right: 16, margin: 0, color: '#a0aec0', fontSize: 11 }}>
      {error || `${matched ? 'Retrieved match connected' : 'Awaiting a retrieved match'} · Illustrative layout; distances do not measure solution quality.`}
    </p>
  </section>;
});

export default GalaxyCanvas;
