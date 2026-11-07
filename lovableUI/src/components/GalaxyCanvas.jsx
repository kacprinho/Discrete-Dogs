import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import * as THREE from 'three';

const SOURCE_ORIGIN = [-5, 0, 0];
const TARGET_ORIGIN = [5, 0, 0];
const validCoords = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) && v.every((n) => Math.abs(n) <= 25);

// Curated anchor concepts for source & target manifolds aligned with corpus.json & engine.py
const DEFAULT_SOURCE_NODES = [
  { id: 'src-traffic', title: 'Traffic Bottleneck', domain: 'distributed_systems', coords: [-4.21, -0.113, 0.735] },
  { id: 'src-orderbook', title: 'Order Book Matching Latency', domain: 'quant_finance', coords: [-5.991, 0.138, 0.185] },
  { id: 'src-dblock', title: 'Database Lock Contention', domain: 'database_concurrency', coords: [-4.626, -0.023, 0.143] },
  { id: 'src-misinfo', title: 'Misinformation Cascade', domain: 'network_security', coords: [-4.577, -0.075, -0.551] },
  { id: 'src-cascade', title: 'Cascading Microservice Failure', domain: 'distributed_systems', coords: [-5.4, 0.18, -0.8] },
  { id: 'src-cache', title: 'Cache Thundering Herd', domain: 'database_concurrency', coords: [-4.8, -0.15, 1.1] },
];

const DEFAULT_TARGET_NODES = [
  { id: 'bio-ant-pheromone', title: 'Ant Colony Pheromone Trails', domain: 'natural', coords: [5.565, -0.106, 0.809], score: 0.82 },
  { id: 'phys-navier-stokes-orderbook', title: 'Navier-Stokes Order Book Viscosity', domain: 'physics', coords: [3.625, 0.135, -0.49], score: 0.77 },
  { id: 'bio-allosteric-enzymes', title: 'Allosteric Enzyme Cooperativity', domain: 'natural', coords: [5.73, -0.053, 0.384], score: 0.79 },
  { id: 'soc-sir-ring-inoculation', title: 'SIR Ring Inoculation', domain: 'social', coords: [4.801, -0.191, -0.644], score: 0.78 },
  { id: 'phys-kuramoto-synchronization', title: 'Kuramoto Phase Synchronization', domain: 'physics', coords: [6.1, 0.08, -0.35], score: 0.74 },
  { id: 'bio-physarum-slime-mold', title: 'Slime Mold Tube Dilation', domain: 'natural', coords: [4.3, 0.15, 0.95], score: 0.72 },
  { id: 'phys-kolmogorov-turbulence', title: 'Kolmogorov Energy Cascade', domain: 'physics', coords: [3.9, -0.12, -0.85], score: 0.71 },
  { id: 'math-sinkhorn-optimal-transport', title: 'Sinkhorn Optimal Transport', domain: 'math', coords: [5.2, 0.1, 1.25], score: 0.75 },
  { id: 'cog-synaptic-pruning', title: 'Synaptic Pruning Microglia', domain: 'cognitive', coords: [6.25, -0.14, 0.45], score: 0.73 },
  { id: 'bio-mycelial-nutrient-trading', title: 'Mycelial Nutrient Trading', domain: 'natural', coords: [4.95, -0.08, 0.22], score: 0.76 },
];

const DOMAIN_COLORS = {
  natural: 0x34d399, // emerald
  physics: 0x38bdf8, // sky blue
  social: 0xf43f5e,  // rose
  math: 0xa855f7,    // violet
  cognitive: 0xf59e0b, // amber
  distributed_systems: 0x22d3ee, // cyan
  quant_finance: 0x38bdf8,
  database_concurrency: 0x06b6d4,
  network_security: 0xec4899,
};

const GalaxyCanvas = forwardRef(function GalaxyCanvas(
  {
    className = '',
    sourceLabel = 'Source domain',
    targetLabel = 'Target domain',
    result = null,
    onNodeSelect = null,
  },
  ref
) {
  const host = useRef(null);
  const engine = useRef(null);
  const [error, setError] = useState('');
  const [matched, setMatched] = useState(false);
  const [tooltip, setTooltip] = useState(null);

  useImperativeHandle(
    ref,
    () => ({
      teleportCamera(targetCoords, sourceCoords = SOURCE_ORIGIN, meta = null) {
        if (!validCoords(targetCoords) || !validCoords(sourceCoords) || !engine.current) return false;
        engine.current.teleport(targetCoords, sourceCoords, meta);
        setMatched(true);
        return true;
      },
      reset() {
        engine.current?.reset();
        setMatched(false);
        setTooltip(null);
      },
    }),
    []
  );

  useEffect(() => {
    const container = host.current;
    if (!container) return;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
        powerPreference: 'high-performance',
      });
    } catch {
      setError('WebGL unavailable. Source and target results remain available in the terminal.');
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 180);
    const look = new THREE.Vector3();
    const desiredLook = new THREE.Vector3(0, 0, 0);
    const baseCameraPos = new THREE.Vector3(0, 6.2, 22);
    const desiredCamera = baseCameraPos.clone();
    camera.position.copy(desiredCamera);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x050814, 1);
    renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;cursor:default;';
    renderer.domElement.setAttribute('aria-hidden', 'true');
    container.appendChild(renderer.domElement);

    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const resources = [];
    const track = (r) => {
      resources.push(r);
      return r;
    };

    // Deterministic PRNG
    let seed = 92837;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    // ==========================================
    // 1. PARTICLE SHADER (ENHANCED POINT SIZE & GLOW)
    // ==========================================
    const particleShaderMaterial = track(
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexColors: true,
        uniforms: {
          uTime: { value: 0 },
        },
        vertexShader: `
          uniform float uTime;
          attribute float aSize;
          varying vec3 tint;
          varying float vSize;
          void main() {
            tint = color;
            vSize = aSize;
            vec4 p = modelViewMatrix * vec4(position, 1.0);
            // Dynamic scale: 155.0 base scaling clamped between 2.5px and 22.0px for vibrant visibility
            gl_PointSize = clamp((aSize * 155.0) / -p.z, 2.5, 22.0);
            gl_Position = projectionMatrix * p;
          }
        `,
        fragmentShader: `
          varying vec3 tint;
          varying float vSize;
          void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord) * 2.0;
            if (dist > 1.0) discard;
            // Soft luminous gaussian-like falloff with white-hot core
            float core = clamp(1.0 - dist * 2.2, 0.0, 1.0);
            float halo = pow(1.0 - dist, 1.8);
            float alpha = clamp(core * 0.85 + halo * 0.55, 0.0, 1.0);
            vec3 rgb = mix(tint, vec3(1.0), core * 0.55);
            gl_FragColor = vec4(rgb, alpha * 0.9);
          }
        `,
      })
    );

    // Populate galaxy discs with 4,800 particles total
    for (const [center, hex] of [
      [SOURCE_ORIGIN, 0x22d3ee],
      [TARGET_ORIGIN, 0xfbbf24],
    ]) {
      const count = 2400;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      const sizes = new Float32Array(count);
      const color = new THREE.Color(hex);

      for (let i = 0; i < count; i++) {
        const radius = Math.sqrt(random()) * 3.4;
        const angle = radius * 1.7 + ((i % 3) * Math.PI * 2) / 3 + (random() - 0.5) * 0.8;
        positions.set(
          [center[0] + Math.cos(angle) * radius, (random() - 0.5) * 0.65, Math.sin(angle) * radius],
          i * 3
        );

        const intensity = 0.4 + random() * 0.6;
        colors.set([color.r * intensity, color.g * intensity, color.b * intensity], i * 3);
        sizes[i] = 0.65 + random() * 0.85; // individual star scale
      }

      const geometry = track(new THREE.BufferGeometry());
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
      scene.add(new THREE.Points(geometry, particleShaderMaterial));
    }

    // ==========================================
    // 2. 3D NODE GEOMETRY & WIREFRAME HALOS
    // ==========================================
    const nodesGroup = new THREE.Group();
    scene.add(nodesGroup);

    // Shared reusable geometries
    const baseSphereGeo = track(new THREE.SphereGeometry(0.24, 24, 24));
    const baseHaloGeo = track(new THREE.SphereGeometry(0.40, 16, 16));
    const baseRingGeo = track(new THREE.RingGeometry(0.35, 0.48, 24));

    const heroSphereGeo = track(new THREE.SphereGeometry(0.42, 32, 32));
    const heroHaloGeo = track(new THREE.SphereGeometry(0.68, 20, 20));
    const heroRingGeo = track(new THREE.RingGeometry(0.55, 0.78, 32));

    const interactiveNodes = [];

    function createNode(meta, isHero = false, customColor = null) {
      const group = new THREE.Group();
      group.position.set(...meta.coords);

      const colorHex = customColor || DOMAIN_COLORS[meta.domain] || (meta.coords[0] < 0 ? 0x22d3ee : 0xfbbf24);
      const color = new THREE.Color(colorHex);

      // Core Solid Sphere
      const sGeo = isHero ? heroSphereGeo : baseSphereGeo;
      const coreMat = track(
        new THREE.MeshBasicMaterial({
          color: color,
        })
      );
      const coreMesh = new THREE.Mesh(sGeo, coreMat);
      group.add(coreMesh);

      // Outer Wireframe Halo
      const hGeo = isHero ? heroHaloGeo : baseHaloGeo;
      const haloMat = track(
        new THREE.MeshBasicMaterial({
          color: color,
          wireframe: true,
          transparent: true,
          opacity: isHero ? 0.6 : 0.38,
          blending: THREE.AdditiveBlending,
        })
      );
      const haloMesh = new THREE.Mesh(hGeo, haloMat);
      group.add(haloMesh);

      // Glowing Orbital Ring
      const rGeo = isHero ? heroRingGeo : baseRingGeo;
      const ringMat = track(
        new THREE.MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: isHero ? 0.5 : 0.28,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
        })
      );
      const ringMesh = new THREE.Mesh(rGeo, ringMat);
      ringMesh.rotation.x = Math.PI / 2;
      group.add(ringMesh);

      nodesGroup.add(group);

      const nodeObj = {
        meta,
        group,
        coreMesh,
        haloMesh,
        ringMesh,
        haloMat,
        ringMat,
        colorHex,
        isHero,
        currentScale: 1.0,
        targetScale: 1.0,
      };

      // Add to raycast targets
      coreMesh.userData = { nodeObj };
      haloMesh.userData = { nodeObj };
      interactiveNodes.push(nodeObj);
      return nodeObj;
    }

    // Populate initial concept nodes
    DEFAULT_SOURCE_NODES.forEach((node) => createNode(node, false));
    DEFAULT_TARGET_NODES.forEach((node) => createNode(node, false));

    // Hero Active Source and Target nodes
    const activeSourceNode = createNode(
      { id: 'active-source', title: 'Problem Bottleneck', coords: SOURCE_ORIGIN },
      true,
      0x22d3ee
    );
    activeSourceNode.group.visible = false;

    const activeTargetNode = createNode(
      { id: 'active-target', title: 'Analogue Match', coords: TARGET_ORIGIN },
      true,
      0xfbbf24
    );
    activeTargetNode.group.visible = false;

    // ==========================================
    // 3. GLOWING TELEPORT BRIDGE & ENERGY PULSE
    // ==========================================
    const bridge = new THREE.Group();
    bridge.visible = false;
    scene.add(bridge);

    let activeCurve = null;
    let bridgeGeometry = null;
    const bridgeMaterials = [0.08, 0.28, 0.85].map((opacity) =>
      track(
        new THREE.MeshBasicMaterial({
          color: 0x8ef8ed,
          transparent: true,
          opacity,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        })
      )
    );

    // Traveling Energy Pulse Mesh along bridge
    const pulseGeo = track(new THREE.SphereGeometry(0.26, 16, 16));
    const pulseMat = track(
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
      })
    );
    const pulseMesh = new THREE.Mesh(pulseGeo, pulseMat);
    pulseMesh.visible = false;
    scene.add(pulseMesh);

    function clearBridge() {
      for (const child of [...bridge.children]) {
        child.geometry?.dispose();
        bridge.remove(child);
      }
      bridgeGeometry = null;
      activeCurve = null;
      pulseMesh.visible = false;
    }

    // ==========================================
    // 4. ENGINE TELEPORT & CAMERA CONTROL
    // ==========================================
    let activeMatchNode = null;

    engine.current = {
      teleport(target, source, meta) {
        clearBridge();
        const start = new THREE.Vector3(...source);
        const end = new THREE.Vector3(...target);

        // Update active hero nodes
        activeSourceNode.group.position.copy(start);
        activeSourceNode.group.visible = true;
        activeSourceNode.meta = {
          ...activeSourceNode.meta,
          title: meta?.sourceTitle || sourceLabel || 'Source Bottleneck',
          coords: source,
        };

        activeTargetNode.group.position.copy(end);
        activeTargetNode.group.visible = true;
        activeTargetNode.meta = {
          ...activeTargetNode.meta,
          title: meta?.targetTitle || targetLabel || 'Target Analogue',
          domain: meta?.domain,
          score: meta?.score,
          coords: target,
        };
        activeMatchNode = activeTargetNode;

        // Build elegant 3-layer Bezier Bridge
        const arc = Math.max(2.4, start.distanceTo(end) * 0.32);
        activeCurve = new THREE.CubicBezierCurve3(
          start,
          start.clone().add(new THREE.Vector3(0, arc, 0)),
          end.clone().add(new THREE.Vector3(0, arc, 0)),
          end
        );

        [0.16, 0.075, 0.022].forEach((radius, i) => {
          bridgeGeometry = new THREE.TubeGeometry(activeCurve, 84, radius, 8, false);
          bridge.add(new THREE.Mesh(bridgeGeometry, bridgeMaterials[i]));
        });
        bridge.visible = true;
        pulseMesh.visible = true;

        // Butter-smooth camera framing transition
        desiredLook.copy(end);
        desiredCamera.set(end.x * 0.55, Math.max(3.8, end.y + 4.2), Math.max(10.5, end.distanceTo(start) * 0.72));
      },
      reset() {
        clearBridge();
        activeSourceNode.group.visible = false;
        activeTargetNode.group.visible = false;
        activeMatchNode = null;
        desiredLook.set(0, 0, 0);
        desiredCamera.copy(baseCameraPos);
      },
    };

    // ==========================================
    // 5. INTERACTION (HOVER RAYCAST & DRAG ORBIT)
    // ==========================================
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(-999, -999);
    let hoveredNode = null;
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    let orbitDelta = { x: 0, y: 0 };
    let targetOrbit = { x: 0, y: 0 };

    const getPointerPos = (e) => {
      const rect = container.getBoundingClientRect();
      return {
        x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
        clientX: e.clientX - rect.left,
        clientY: e.clientY - rect.top,
      };
    };

    const handlePointerDown = (e) => {
      isDragging = true;
      dragStart = { x: e.clientX, y: e.clientY };
    };

    const handlePointerMove = (e) => {
      const p = getPointerPos(e);
      pointer.x = p.x;
      pointer.y = p.y;

      if (isDragging) {
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;
        dragStart = { x: e.clientX, y: e.clientY };
        targetOrbit.x += dx * 0.005;
        targetOrbit.y = Math.max(-0.55, Math.min(0.55, targetOrbit.y + dy * 0.005));
      }
    };

    const handlePointerUp = () => {
      isDragging = false;
    };

    const handlePointerLeave = () => {
      isDragging = false;
      pointer.x = -999;
      pointer.y = -999;
      hoveredNode = null;
      setTooltip(null);
      renderer.domElement.style.cursor = 'default';
    };

    const handleClick = () => {
      if (hoveredNode && onNodeSelect) {
        onNodeSelect(hoveredNode.meta);
      }
    };

    const domEl = renderer.domElement;
    domEl.addEventListener('pointerdown', handlePointerDown);
    domEl.addEventListener('pointermove', handlePointerMove);
    domEl.addEventListener('pointerup', handlePointerUp);
    domEl.addEventListener('pointerleave', handlePointerLeave);
    domEl.addEventListener('click', handleClick);

    // ==========================================
    // 6. RESIZE OBSERVER
    // ==========================================
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

    // ==========================================
    // 7. ANIMATION RENDER LOOP (BUTTER-SMOOTH LERP)
    // ==========================================
    let frame = 0;
    let previous = performance.now();
    let disposed = false;

    function render(now) {
      if (disposed) return;
      const dt = Math.min((now - previous) / 1000, 0.1);
      previous = now;

      // Update particle shader clock
      particleShaderMaterial.uniforms.uTime.value = now * 0.001;

      // Smooth Orbit Damping
      orbitDelta.x = THREE.MathUtils.lerp(orbitDelta.x, targetOrbit.x, 0.1);
      orbitDelta.y = THREE.MathUtils.lerp(orbitDelta.y, targetOrbit.y, 0.1);

      // Butter-Smooth Camera Position & LookAt Lerp
      const alpha = motion.matches ? 1 : 1 - Math.exp(-3.5 * dt);
      const effectiveDesiredCam = desiredCamera.clone();
      // Apply gentle interactive orbit tilt
      effectiveDesiredCam.x += orbitDelta.x * 6.0;
      effectiveDesiredCam.y += orbitDelta.y * 4.0;

      camera.position.lerp(effectiveDesiredCam, alpha);
      look.lerp(desiredLook, alpha);
      camera.lookAt(look);

      // Raycasting for interactive hover
      raycaster.setFromCamera(pointer, camera);
      const hitCandidates = interactiveNodes.map((n) => n.coreMesh);
      const hits = raycaster.intersectObjects(hitCandidates, false);

      if (hits.length > 0) {
        const hitNode = hits[0].object.userData?.nodeObj;
        if (hitNode && hitNode !== hoveredNode) {
          hoveredNode = hitNode;
          domEl.style.cursor = 'pointer';

          // Project node coords into screen space for padded tooltip badge
          const wp = hitNode.group.position.clone();
          wp.project(camera);
          const hw = container.clientWidth / 2;
          const hh = container.clientHeight / 2;
          const screenX = wp.x * hw + hw;
          const screenY = -wp.y * hh + hh;

          setTooltip({
            title: hitNode.meta.title,
            domain: hitNode.meta.domain,
            score: hitNode.meta.score,
            coords: hitNode.meta.coords,
            x: Math.round(screenX),
            y: Math.round(screenY),
            color: `#${hitNode.colorHex.toString(16).padStart(6, '0')}`,
            glow: `rgba(${hitNode.colorHex >> 16}, ${(hitNode.colorHex >> 8) & 255}, ${hitNode.colorHex & 255}, 0.4)`,
          });
        }
      } else if (hoveredNode) {
        hoveredNode = null;
        setTooltip(null);
        domEl.style.cursor = 'default';
      }

      // ==========================================
      // DYNAMIC SCALING ON ZOOM / DISTANCE & HOVER
      // ==========================================
      const camPos = camera.position;
      const time = now * 0.001;

      interactiveNodes.forEach((node) => {
        // Distance scaling factor keeps nodes legible and visually striking at all camera distances
        const distToCam = camPos.distanceTo(node.group.position);
        const zoomFactor = THREE.MathUtils.clamp(distToCam / 14.0, 0.85, 1.55);

        // Hover scale boost
        const isHovered = node === hoveredNode;
        const isHeroTarget = node === activeMatchNode;
        const hoverMultiplier = isHovered ? 1.55 : isHeroTarget ? 1.25 : 1.0;

        node.targetScale = hoverMultiplier * zoomFactor;
        node.currentScale = THREE.MathUtils.lerp(node.currentScale, node.targetScale, 0.14);
        node.group.scale.setScalar(node.currentScale);

        // Subtle slow orbital rotation of wireframe halo and ring
        node.haloMesh.rotation.y += 0.012;
        node.haloMesh.rotation.x += 0.006;
        node.ringMesh.rotation.z += 0.008;

        // Pulsing halo for heroes or hovered nodes
        if (node.isHero || isHovered) {
          const pulse = 0.45 + 0.3 * Math.sin(time * 3.5);
          node.haloMat.opacity = pulse;
        }
      });

      // Animate Traveling Bridge Energy Pulse
      if (activeCurve && pulseMesh.visible) {
        const pulseT = (time * 0.55) % 1.0;
        const pulsePos = activeCurve.getPoint(pulseT);
        pulseMesh.position.copy(pulsePos);
        const pulseScale = 1.0 + 0.35 * Math.sin(pulseT * Math.PI);
        pulseMesh.scale.setScalar(pulseScale);
      }

      if (!document.hidden) renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    }

    frame = requestAnimationFrame(render);

    const lost = (event) => {
      event.preventDefault();
      cancelAnimationFrame(frame);
      setError('Graphics context lost. Reload to restore the visualization.');
    };
    domEl.addEventListener('webglcontextlost', lost);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      domEl.removeEventListener('pointerdown', handlePointerDown);
      domEl.removeEventListener('pointermove', handlePointerMove);
      domEl.removeEventListener('pointerup', handlePointerUp);
      domEl.removeEventListener('pointerleave', handlePointerLeave);
      domEl.removeEventListener('click', handleClick);
      domEl.removeEventListener('webglcontextlost', lost);
      clearBridge();
      resources.forEach((r) => r.dispose());
      renderer.dispose();
      domEl.remove();
      engine.current = null;
    };
  }, [onNodeSelect, sourceLabel, targetLabel]);

  return (
    <section
      className={className}
      aria-label="Cross-domain galaxy visualization"
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        minHeight: 320,
        overflow: 'hidden',
        background: '#050814',
      }}
    >
      <div ref={host} style={{ position: 'absolute', inset: 0 }} />

      {/* TOP DOMAIN BADGES WITH GENEROUS LABEL PADDING */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 16,
          right: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 16,
          pointerEvents: 'none',
          zIndex: 10,
        }}
      >
        {/* Source Domain Badge */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 10,
            padding: '8px 18px',
            background: 'rgba(6, 18, 36, 0.88)',
            border: '1px solid rgba(34, 211, 238, 0.45)',
            borderRadius: '9999px',
            boxShadow: '0 0 20px rgba(34, 211, 238, 0.2), inset 0 0 10px rgba(34, 211, 238, 0.08)',
            backdropFilter: 'blur(10px)',
            color: '#22d3ee',
            letterSpacing: '0.08em',
            fontWeight: 600,
            textTransform: 'uppercase',
            fontSize: 11,
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: '#22d3ee',
              boxShadow: '0 0 8px #22d3ee',
            }}
          />
          <span style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {sourceLabel}
          </span>
        </div>

        {/* Target Domain Badge */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 10,
            padding: '8px 18px',
            background: 'rgba(36, 24, 6, 0.88)',
            border: '1px solid rgba(251, 191, 36, 0.45)',
            borderRadius: '9999px',
            boxShadow: '0 0 20px rgba(251, 191, 36, 0.2), inset 0 0 10px rgba(251, 191, 36, 0.08)',
            backdropFilter: 'blur(10px)',
            color: '#fbbf24',
            letterSpacing: '0.08em',
            fontWeight: 600,
            textTransform: 'uppercase',
            fontSize: 11,
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: '#fbbf24',
              boxShadow: '0 0 8px #fbbf24',
            }}
          />
          <span style={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {targetLabel}
          </span>
        </div>
      </div>

      {/* DYNAMIC FLOATING NODE TOOLTIP ANCHORED IN 3D SPACE */}
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: `${tooltip.x}px`,
            top: `${tooltip.y}px`,
            transform: 'translate(-50%, -125%)',
            pointerEvents: 'none',
            zIndex: 30,
            padding: '10px 16px',
            background: 'rgba(6, 10, 24, 0.94)',
            border: `1px solid ${tooltip.color || 'rgba(34, 211, 238, 0.6)'}`,
            borderRadius: '10px',
            boxShadow: `0 8px 24px rgba(0,0,0,0.85), 0 0 18px ${tooltip.glow || 'rgba(34, 211, 238, 0.35)'}`,
            backdropFilter: 'blur(12px)',
            minWidth: 180,
            maxWidth: 290,
            transition: 'transform 0.06s ease-out',
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc', marginBottom: 4, lineHeight: 1.3 }}>
            {tooltip.title}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            {tooltip.domain && (
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  background: 'rgba(255, 255, 255, 0.08)',
                  color: tooltip.color || '#38bdf8',
                }}
              >
                {tooltip.domain}
              </span>
            )}
            {tooltip.score !== undefined && (
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: '4px',
                  background: 'rgba(52, 211, 153, 0.15)',
                  color: '#34d399',
                  border: '1px solid rgba(52, 211, 153, 0.3)',
                }}
              >
                SIM {(tooltip.score * 100).toFixed(1)}%
              </span>
            )}
          </div>
          {tooltip.coords && (
            <div style={{ fontSize: 9, color: '#64748b', marginTop: 4, fontFamily: 'monospace' }}>
              [{tooltip.coords.map((n) => n.toFixed(2)).join(', ')}]
            </div>
          )}
        </div>
      )}

      {/* BOTTOM STATUS BAR WITH PADDED PILL BADGES */}
      <div
        style={{
          position: 'absolute',
          bottom: 14,
          left: 16,
          right: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          pointerEvents: 'none',
          zIndex: 10,
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 14px',
            background: 'rgba(5, 8, 20, 0.85)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '9999px',
            backdropFilter: 'blur(8px)',
            color: '#94a3b8',
            fontSize: 11,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: matched ? '#34d399' : '#38bdf8',
              boxShadow: matched ? '0 0 6px #34d399' : '0 0 6px #38bdf8',
            }}
          />
          <span>
            {error ||
              (matched
                ? 'Latent teleportation bridge active · Concept aligned'
                : 'Awaiting bottleneck query · 3D manifold ready')}
          </span>
        </div>
        <div
          style={{
            padding: '4px 10px',
            background: 'rgba(5, 8, 20, 0.65)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '6px',
            color: '#64748b',
            fontSize: 10,
            letterSpacing: '0.05em',
          }}
        >
          DRAG TO ORBIT · HOVER TO INSPECT
        </div>
      </div>
    </section>
  );
});

export default GalaxyCanvas;
