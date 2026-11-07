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
  { id: 'bio-ant-pheromone', title: 'Ant Colony Pheromone Trails', domain: 'natural', coords: [5.565, -0.106, 0.809] },
  { id: 'phys-navier-stokes-orderbook', title: 'Navier-Stokes Order Book Viscosity', domain: 'physics', coords: [3.625, 0.135, -0.49] },
  { id: 'bio-allosteric-enzymes', title: 'Allosteric Enzyme Cooperativity', domain: 'natural', coords: [5.73, -0.053, 0.384] },
  { id: 'soc-sir-ring-inoculation', title: 'SIR Ring Inoculation', domain: 'social', coords: [4.801, -0.191, -0.644] },
  { id: 'phys-kuramoto-synchronization', title: 'Kuramoto Phase Synchronization', domain: 'physics', coords: [6.1, 0.08, -0.35] },
  { id: 'bio-physarum-slime-mold', title: 'Slime Mold Tube Dilation', domain: 'natural', coords: [4.3, 0.15, 0.95] },
  { id: 'phys-kolmogorov-turbulence', title: 'Kolmogorov Energy Cascade', domain: 'physics', coords: [3.9, -0.12, -0.85] },
  { id: 'math-sinkhorn-optimal-transport', title: 'Sinkhorn Optimal Transport', domain: 'math', coords: [5.2, 0.1, 1.25] },
  { id: 'cog-synaptic-pruning', title: 'Synaptic Pruning Microglia', domain: 'cognitive', coords: [6.25, -0.14, 0.45] },
  { id: 'bio-mycelial-nutrient-trading', title: 'Mycelial Nutrient Trading', domain: 'natural', coords: [4.95, -0.08, 0.22] },
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
  const latest = useRef({ sourceLabel, targetLabel, onNodeSelect });
  latest.current = { sourceLabel, targetLabel, onNodeSelect };
  const appliedResult = useRef(null);
  const [error, setError] = useState('');
  const [matched, setMatched] = useState(false);

  // Direct DOM references for 60fps hardware-accelerated badge & tooltip positioning without React re-render stutter
  const tooltipRef = useRef(null);
  const tooltipTitleRef = useRef(null);
  const tooltipDomainRef = useRef(null);
  const tooltipScoreRef = useRef(null);
  const tooltipCoordsRef = useRef(null);

  const heroBadgeRef = useRef(null);
  const heroTitleRef = useRef(null);
  const heroMetaRef = useRef(null);

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
            // Sizing specification: scaled point size clamped between 3.0px and 28.0px for vivid visibility
            gl_PointSize = clamp((aSize * 220.0) / -p.z, 3.0, 28.0);
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
            // Multi-tier luminous gaussian-like falloff with white-hot core
            float core = clamp(1.0 - dist * 2.0, 0.0, 1.0);
            float halo = pow(1.0 - dist, 1.6);
            float alpha = clamp(core * 0.9 + halo * 0.6, 0.0, 1.0);
            vec3 rgb = mix(tint, vec3(1.0), core * 0.65);
            gl_FragColor = vec4(rgb, alpha * 0.95);
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

    // Substantial node geometry scale matching sizing specifications
    const baseSphereGeo = track(new THREE.SphereGeometry(0.38, 24, 24));
    const baseHaloGeo = track(new THREE.SphereGeometry(0.65, 18, 18));
    const baseRingGeo = track(new THREE.RingGeometry(0.52, 0.78, 28));

    const heroSphereGeo = track(new THREE.SphereGeometry(0.60, 32, 32));
    const heroHaloGeo = track(new THREE.SphereGeometry(1.05, 22, 22));
    const heroRingGeo = track(new THREE.RingGeometry(0.85, 1.25, 36));

    // Generous hit geometries for effortless raycast hover detection
    const baseHitGeo = track(new THREE.SphereGeometry(0.80, 8, 8));
    const heroHitGeo = track(new THREE.SphereGeometry(1.25, 8, 8));

    const interactiveNodes = [];
    const dynamicCandidateNodes = [];

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
          opacity: isHero ? 0.65 : 0.40,
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
          opacity: isHero ? 0.55 : 0.30,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
        })
      );
      const ringMesh = new THREE.Mesh(rGeo, ringMat);
      ringMesh.rotation.x = Math.PI / 2;
      group.add(ringMesh);

      // Invisible hit sphere for generous hover raycasting
      const hitGeo = isHero ? heroHitGeo : baseHitGeo;
      const hitMesh = new THREE.Mesh(
        hitGeo,
        track(new THREE.MeshBasicMaterial({ visible: false, wireframe: false }))
      );
      group.add(hitMesh);

      nodesGroup.add(group);

      const nodeObj = {
        meta,
        group,
        coreMesh,
        haloMesh,
        ringMesh,
        hitMesh,
        haloMat,
        ringMat,
        colorHex,
        isHero,
        currentScale: 1.0,
        targetScale: 1.0,
      };

      // Set user data for raycaster
      hitMesh.userData = { nodeObj };
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
    const pulseGeo = track(new THREE.SphereGeometry(0.32, 16, 16));
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
    function clearCandidates() {
      for (const node of dynamicCandidateNodes) {
        nodesGroup.remove(node.group);
        const index = interactiveNodes.indexOf(node);
        if (index !== -1) interactiveNodes.splice(index, 1);
        node.group.traverse((object) => {
          if (!object.material) return;
          object.material.dispose();
          const resourceIndex = resources.indexOf(object.material);
          if (resourceIndex !== -1) resources.splice(resourceIndex, 1);
        });
      }
      dynamicCandidateNodes.length = 0;
      hoveredNode = null;
      if (tooltipRef.current) tooltipRef.current.style.display = 'none';
    }
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
          title: meta?.sourceTitle || latest.current.sourceLabel || 'Source Bottleneck',
          coords: source,
        };

        activeTargetNode.group.position.copy(end);
        activeTargetNode.group.visible = true;
        activeTargetNode.meta = {
          ...activeTargetNode.meta,
          title: meta?.targetTitle || latest.current.targetLabel || 'Target Analogue',
          domain: meta?.domain,
          score: meta?.score,
          coords: target,
        };
        activeMatchNode = activeTargetNode;

        // Clear previous dynamic candidate nodes if any
        clearCandidates();

        // If meta provides candidates, instantiate them dynamically in 3D space
        if (Array.isArray(meta?.candidates)) {
          meta.candidates.forEach((cand) => {
            const pt = cand.point || cand.coords || cand.target_point;
            if (validCoords(pt)) {
              // Skip if coords match active target exactly
              const d = Math.hypot(pt[0] - target[0], pt[1] - target[1], pt[2] - target[2]);
              if (d > 0.05) {
                const candNode = createNode(
                  {
                    id: cand.id || cand.title,
                    title: cand.title,
                    domain: cand.domain,
                    score: cand.score,
                    coords: pt,
                  },
                  false
                );
                dynamicCandidateNodes.push(candNode);
              }
            }
          });
        }

        // Update hero badge texts
        if (heroTitleRef.current) {
          heroTitleRef.current.textContent = activeTargetNode.meta.title;
        }
        if (heroMetaRef.current) {
          const domainStr = (meta?.domain || '').toUpperCase();
          const scoreStr = Number.isFinite(meta?.score) ? ` · SIM ${(meta.score * 100).toFixed(1)}%` : '';
          heroMetaRef.current.textContent = `${domainStr || 'ANALOGUE'}${scoreStr}`;
        }

        // Build elegant 3-layer Bezier Bridge
        const arc = Math.max(2.4, start.distanceTo(end) * 0.32);
        activeCurve = new THREE.CubicBezierCurve3(
          start,
          start.clone().add(new THREE.Vector3(0, arc, 0)),
          end.clone().add(new THREE.Vector3(0, arc, 0)),
          end
        );

        [0.18, 0.085, 0.024].forEach((radius, i) => {
          bridgeGeometry = new THREE.TubeGeometry(activeCurve, 84, radius, 8, false);
          bridge.add(new THREE.Mesh(bridgeGeometry, bridgeMaterials[i]));
        });
        bridge.visible = true;
        pulseMesh.visible = true;

        // Butter-smooth camera framing transition
        desiredLook.copy(end);
        const dist = start.distanceTo(end);
        desiredCamera.set(
          end.x * 0.45 + (start.x + end.x) * 0.1,
          Math.max(4.2, end.y + 4.5),
          Math.max(11.0, dist * 0.72)
        );
      },
      reset() {
        clearBridge();
        activeSourceNode.group.visible = false;
        activeTargetNode.group.visible = false;
        activeMatchNode = null;

        clearCandidates();

        if (heroBadgeRef.current) heroBadgeRef.current.style.display = 'none';
        if (tooltipRef.current) tooltipRef.current.style.display = 'none';

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

    const handleWheel = (event) => {
      event.preventDefault();
      const offset = desiredCamera.clone().sub(desiredLook);
      const distance = THREE.MathUtils.clamp(offset.length() * Math.exp(THREE.MathUtils.clamp(event.deltaY, -100, 100) * 0.002), 7, 35);
      desiredCamera.copy(desiredLook).add(offset.setLength(distance));
    };

    const handlePointerLeave = () => {
      isDragging = false;
      pointer.x = -999;
      pointer.y = -999;
      hoveredNode = null;
      if (tooltipRef.current) tooltipRef.current.style.display = 'none';
      renderer.domElement.style.cursor = 'default';
    };

    const handleClick = () => {
      if (hoveredNode && latest.current.onNodeSelect) {
        latest.current.onNodeSelect(hoveredNode.meta);
      }
    };

    const domEl = renderer.domElement;
    domEl.addEventListener('pointerdown', handlePointerDown);
    domEl.addEventListener('pointermove', handlePointerMove);
    domEl.addEventListener('pointerup', handlePointerUp);
    domEl.addEventListener('pointerleave', handlePointerLeave);
    domEl.addEventListener('click', handleClick);
    domEl.addEventListener('wheel', handleWheel, { passive: false });

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
      const orbitAlpha = motion.matches ? 1 : 1 - Math.exp(-6.32 * dt);
      orbitDelta.x = THREE.MathUtils.lerp(orbitDelta.x, targetOrbit.x, orbitAlpha);
      orbitDelta.y = THREE.MathUtils.lerp(orbitDelta.y, targetOrbit.y, orbitAlpha);

      // Butter-Smooth Camera Position & LookAt Lerp (framerate-independent exponential smoothing)
      const alpha = motion.matches ? 1 : 1 - Math.exp(-3.8 * dt);
      const effectiveDesiredCam = desiredCamera.clone();
      // Apply gentle interactive orbit tilt
      effectiveDesiredCam.x += orbitDelta.x * 6.0;
      effectiveDesiredCam.y += orbitDelta.y * 4.0;

      camera.position.lerp(effectiveDesiredCam, alpha);
      look.lerp(desiredLook, alpha);
      camera.lookAt(look);

      // Raycasting for interactive hover across all hit meshes
      raycaster.setFromCamera(pointer, camera);
      scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      const hitCandidates = interactiveNodes.filter((n) => n.group.visible).map((n) => n.hitMesh || n.coreMesh);
      const hits = raycaster.intersectObjects(hitCandidates, false);

      if (hits.length > 0) {
        const hitNode = hits[0].object.userData?.nodeObj;
        if (hitNode) {
          if (hitNode !== hoveredNode) {
            hoveredNode = hitNode;
            domEl.style.cursor = 'pointer';

            // Populate tooltip text
            if (tooltipTitleRef.current) {
              tooltipTitleRef.current.textContent = hitNode.meta.title;
            }
            if (tooltipDomainRef.current) {
              tooltipDomainRef.current.textContent = hitNode.meta.domain || 'CONCEPT';
              tooltipDomainRef.current.style.color = `#${hitNode.colorHex.toString(16).padStart(6, '0')}`;
            }
            if (tooltipScoreRef.current) {
              if (Number.isFinite(hitNode.meta.score)) {
                tooltipScoreRef.current.textContent = `SIM ${(hitNode.meta.score * 100).toFixed(1)}%`;
                tooltipScoreRef.current.style.display = 'inline-block';
              } else {
                tooltipScoreRef.current.style.display = 'none';
              }
            }
            if (tooltipCoordsRef.current && hitNode.meta.coords) {
              tooltipCoordsRef.current.textContent = `[${hitNode.meta.coords.map((n) => n.toFixed(2)).join(', ')}]`;
            }
          }
        }
      } else if (hoveredNode) {
        hoveredNode = null;
        domEl.style.cursor = 'default';
        if (tooltipRef.current) tooltipRef.current.style.display = 'none';
      }

      // Project hovered tooltip position smoothly every frame (follows camera lerp & orbit!)
      if (hoveredNode && tooltipRef.current) {
        const wp = hoveredNode.group.position.clone();
        wp.project(camera);
        if (wp.z < 1) {
          const hw = container.clientWidth / 2;
          const hh = container.clientHeight / 2;
          const screenX = wp.x * hw + hw;
          const screenY = -wp.y * hh + hh;
          tooltipRef.current.style.display = 'block';
          tooltipRef.current.style.transform = `translate3d(${screenX}px, ${screenY}px, 0) translate(-50%, -130%)`;
        } else {
          tooltipRef.current.style.display = 'none';
        }
      }

      // Project active target hero badge in 3D screen space
      if (activeTargetNode.group.visible && heroBadgeRef.current) {
        const wp = activeTargetNode.group.position.clone();
        wp.project(camera);
        if (wp.z < 1) {
          const hw = container.clientWidth / 2;
          const hh = container.clientHeight / 2;
          const screenX = wp.x * hw + hw;
          const screenY = -wp.y * hh + hh;
          heroBadgeRef.current.style.display = 'block';
          heroBadgeRef.current.style.transform = `translate3d(${screenX}px, ${screenY}px, 0) translate(-50%, -145%)`;
        } else {
          heroBadgeRef.current.style.display = 'none';
        }
      } else if (heroBadgeRef.current) {
        heroBadgeRef.current.style.display = 'none';
      }

      // ==========================================
      // DYNAMIC SCALING ON ZOOM / DISTANCE & HOVER
      // ==========================================
      const camPos = camera.position;
      const time = now * 0.001;

      interactiveNodes.forEach((node) => {
        // Distance scaling factor keeps nodes legible and visually striking at all camera distances
        const distToCam = camPos.distanceTo(node.group.position);
        const zoomFactor = THREE.MathUtils.clamp(distToCam / 11.5, 0.85, 1.85);

        // Hover scale boost: 1.6x on hover, 1.35x for active teleport match
        const isHovered = node === hoveredNode;
        const isHeroTarget = node === activeMatchNode || node === activeTargetNode || node === activeSourceNode;
        const hoverMultiplier = isHovered ? 1.6 : isHeroTarget ? 1.35 : 1.0;

        node.targetScale = hoverMultiplier * zoomFactor;
        node.currentScale = THREE.MathUtils.lerp(node.currentScale, node.targetScale, motion.matches ? 1 : 1 - Math.exp(-10.46 * dt));
        node.group.scale.setScalar(node.currentScale);

        // Subtle slow orbital rotation of wireframe halo and ring
        if (!motion.matches) {
          node.haloMesh.rotation.y += 0.72 * dt;
          node.haloMesh.rotation.x += 0.36 * dt;
          node.ringMesh.rotation.z += 0.48 * dt;
        }

        // Pulsing halo for heroes or hovered nodes
        if (node.isHero || isHovered) {
          const pulse = motion.matches ? 0.55 : 0.45 + 0.35 * Math.sin(time * 3.5);
          node.haloMat.opacity = pulse;
        }
      });

      // Animate Traveling Bridge Energy Pulse
      if (activeCurve && pulseMesh.visible && !motion.matches) {
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
      domEl.removeEventListener('wheel', handleWheel);
      domEl.removeEventListener('webglcontextlost', lost);
      clearBridge();
      appliedResult.current = null;
      resources.forEach((r) => r.dispose());
      renderer.dispose();
      domEl.remove();
      engine.current = null;
    };
  }, []);

  // Sync when result prop updates
  useEffect(() => {
    if (!result) { engine.current?.reset(); appliedResult.current = null; setMatched(false); return; }
    if (!engine.current || appliedResult.current === result) return;
    const targetPoint = result.target_point || result.target_coords;
    const sourcePoint = result.source_point || result.source_coords || SOURCE_ORIGIN;
    if (validCoords(targetPoint) && validCoords(sourcePoint)) {
      appliedResult.current = result;
      engine.current.teleport(targetPoint, sourcePoint, {
        targetTitle: result.match?.title,
        domain: result.match?.domain,
        score: result.match?.score,
        candidates: result.candidates,
      });
      setMatched(true);
    }
  }, [result]);


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
            padding: '8px 20px',
            background: 'rgba(6, 18, 36, 0.90)',
            border: '1px solid rgba(34, 211, 238, 0.5)',
            borderRadius: '9999px',
            boxShadow: '0 0 20px rgba(34, 211, 238, 0.22), inset 0 0 12px rgba(34, 211, 238, 0.1)',
            backdropFilter: 'blur(12px)',
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
          <span style={{ maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {sourceLabel}
          </span>
        </div>

        {/* Target Domain Badge */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 10,
            padding: '8px 20px',
            background: 'rgba(36, 24, 6, 0.90)',
            border: '1px solid rgba(251, 191, 36, 0.5)',
            borderRadius: '9999px',
            boxShadow: '0 0 20px rgba(251, 191, 36, 0.22), inset 0 0 12px rgba(251, 191, 36, 0.1)',
            backdropFilter: 'blur(12px)',
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
          <span style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {targetLabel}
          </span>
        </div>
      </div>

      {/* 3D-ANCHORED HERO TARGET BADGE WITH GENEROUS PADDING */}
      <div
        ref={heroBadgeRef}
        style={{
          display: 'none',
          position: 'absolute',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          zIndex: 25,
          padding: '10px 18px',
          background: 'rgba(8, 14, 30, 0.94)',
          border: '1px solid rgba(251, 191, 36, 0.7)',
          borderRadius: '12px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.85), 0 0 24px rgba(251, 191, 36, 0.35)',
          backdropFilter: 'blur(14px)',
          minWidth: 200,
          maxWidth: 320,
          textAlign: 'center',
          willChange: 'transform',
        }}
      >
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: '#fbbf24',
            marginBottom: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#fbbf24',
              boxShadow: '0 0 8px #fbbf24',
            }}
          />
          Target Analogue
        </div>
        <div
          ref={heroTitleRef}
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: '#f8fafc',
            lineHeight: 1.3,
            marginBottom: 6,
          }}
        >
          {targetLabel}
        </div>
        <div
          ref={heroMetaRef}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 10,
            fontWeight: 600,
            padding: '3px 10px',
            borderRadius: '6px',
            background: 'rgba(251, 191, 36, 0.15)',
            border: '1px solid rgba(251, 191, 36, 0.35)',
            color: '#fbbf24',
          }}
        >
          ACTIVE TELEPORT MATCH
        </div>
      </div>

      {/* 3D-ANCHORED HOVER TOOLTIP WITH GENEROUS PADDING */}
      <div
        ref={tooltipRef}
        style={{
          display: 'none',
          position: 'absolute',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          zIndex: 30,
          padding: '12px 18px',
          background: 'rgba(6, 12, 26, 0.95)',
          border: '1px solid rgba(34, 211, 238, 0.65)',
          borderRadius: '12px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.88), 0 0 20px rgba(34, 211, 238, 0.35)',
          backdropFilter: 'blur(14px)',
          minWidth: 200,
          maxWidth: 320,
          willChange: 'transform',
        }}
      >
        <div
          ref={tooltipTitleRef}
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: '#f8fafc',
            marginBottom: 6,
            lineHeight: 1.35,
          }}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span
            ref={tooltipDomainRef}
            style={{
              fontSize: 10,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              padding: '3px 10px',
              borderRadius: '5px',
              background: 'rgba(255, 255, 255, 0.08)',
              color: '#38bdf8',
            }}
          />
          <span
            ref={tooltipScoreRef}
            style={{
              fontSize: 10,
              fontWeight: 700,
              padding: '3px 10px',
              borderRadius: '5px',
              background: 'rgba(52, 211, 153, 0.15)',
              color: '#34d399',
              border: '1px solid rgba(52, 211, 153, 0.3)',
            }}
          />
        </div>
        <div
          ref={tooltipCoordsRef}
          style={{
            fontSize: 9,
            color: '#64748b',
            marginTop: 6,
            fontFamily: 'monospace',
          }}
        />
      </div>

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
            padding: '7px 18px',
            background: 'rgba(5, 8, 20, 0.88)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '9999px',
            backdropFilter: 'blur(10px)',
            color: '#94a3b8',
            fontSize: 11,
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: matched ? '#34d399' : '#38bdf8',
              boxShadow: matched ? '0 0 8px #34d399' : '0 0 8px #38bdf8',
            }}
          />
          <span>
            {error ||
              (matched
                ? 'Retrieved match connected · Illustrative layout'
                : 'Awaiting query · Illustrative concept layout')}
          </span>
        </div>
        <div
          style={{
            padding: '5px 12px',
            background: 'rgba(5, 8, 20, 0.72)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '6px',
            color: '#64748b',
            fontSize: 10,
            letterSpacing: '0.06em',
          }}
        >
          DRAG TO ORBIT · HOVER TO INSPECT
        </div>
      </div>
    </section>
  );
});

export default GalaxyCanvas;
