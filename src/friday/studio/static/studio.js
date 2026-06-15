/*
 * FRIDAY 3D Studio — frontend controller (no build step).
 *
 * Loads Three.js + addons via the page's importmap. Responsibilities:
 *   1. Render: POST /studio/generate, build Three.js meshes from the *validated
 *      JSON Scene* the backend returns (the shared schema). Never executes any
 *      server-sourced code — only maps trusted-shape data to geometry.
 *   2. Hand-tracking: MediaPipe Hands over the webcam → pinch=zoom, open-hand
 *      move=rotate, two-hand spread=scale, index-point=highlight. Degrades
 *      gracefully if MediaPipe or the camera is unavailable.
 *   3. Voice: Web Speech API → "make <thing>" generates; rotate/zoom/reset/
 *      wireframe/color/download run locally.
 *   4. Download: export the live scene to GLB / STL / OBJ.
 *
 * Everything is defensive: a missing API, blocked camera, or absent SpeechRecognition
 * shows a friendly HUD/toast message and never hard-crashes the page.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { STLExporter } from "three/addons/exporters/STLExporter.js";
import { OBJExporter } from "three/addons/exporters/OBJExporter.js";

// ── DOM handles ────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const viewport = $("viewport");
const webcamEl = $("webcam");
const overlayEl = $("overlay");
const promptForm = $("prompt-form");
const promptInput = $("prompt");
const micBtn = $("mic");
const camToggleBtn = $("cam-toggle");
const resetViewBtn = $("reset-view");
const wireToggleBtn = $("wire-toggle");
const connStatus = $("conn-status");
const hud = {
  gesture: $("hud-gesture"),
  voice: $("hud-voice"),
  heard: $("hud-heard"),
  status: $("hud-status"),
};
const toastEl = $("toast");

// ── Small UI helpers ────────────────────────────────────────────────────────
function setHud(key, text) {
  if (hud[key]) hud[key].textContent = text;
}

let toastTimer = null;
function toast(message, kind = "info") {
  if (!toastEl) return;
  toastEl.textContent = message;
  toastEl.className = "";
  if (kind === "warn") toastEl.classList.add("toast-warn");
  if (kind === "error") toastEl.classList.add("toast-error");
  toastEl.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, 4200);
}

function setConn(state, label) {
  if (!connStatus) return;
  connStatus.classList.remove("pill-ok", "pill-bad", "pill-muted");
  connStatus.classList.add(
    state === "ok" ? "pill-ok" : state === "bad" ? "pill-bad" : "pill-muted"
  );
  const labelEl = connStatus.querySelector(".conn-label");
  if (labelEl) labelEl.textContent = label;
}

// ── Three.js scene scaffold ──────────────────────────────────────────────────
const DEFAULT_BG = "#101014";

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(DEFAULT_BG);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
camera.position.set(4, 3, 6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.5, 0);

// Lighting: a soft hemisphere fill + a key directional, so MeshStandardMaterial
// metalness/roughness read correctly.
const hemi = new THREE.HemisphereLight(0xffffff, 0x202028, 0.85);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(5, 8, 6);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 0.4);
rim.position.set(-6, 3, -4);
scene.add(rim);

// Ground grid for spatial reference.
const grid = new THREE.GridHelper(20, 20, 0x3a3a48, 0x23232e);
grid.position.y = 0;
scene.add(grid);

// The model lives under one group so gestures/voice can rotate/scale it as a unit
// without disturbing lights/grid. Rebuilt on each generate.
let modelGroup = new THREE.Group();
scene.add(modelGroup);

// Selection highlight state.
let selected = null;
let selectedOriginalEmissive = null;

// ── Geometry factory: Scene-node "type" → THREE geometry ─────────────────────
// Mirrors the shared contract (abbreviated keys w/h/d/r/tube). The backend
// normalizes real-model output to those keys, but this factory ALSO reads each
// dimension with a full-name synonym fallback so a raw/un-normalized scene (e.g.
// loaded straight from a model that emitted "width"/"radius"/"tubeRadius") still
// renders the requested geometry instead of silently using defaults.
function buildGeometry(type, params) {
  const p = params || {};
  const n = (v, d) => (typeof v === "number" && isFinite(v) ? v : d);
  // Defensive dimension reads: prefer the canonical key, fall back to synonyms.
  const w = n(p.w ?? p.width ?? p.size, 1);
  const h = n(p.h ?? p.height ?? p.length, 1);
  const d = n(p.d ?? p.depth, 1);
  const r = n(p.r ?? p.radius ?? p.radiusTop ?? p.radiusBottom, 0.5);
  const tube = n(p.tube ?? p.tubeRadius ?? p.tube_radius, 0.18);
  switch (type) {
    case "box":
      return new THREE.BoxGeometry(w, h, d);
    case "sphere":
      return new THREE.SphereGeometry(r, 32, 24);
    case "cylinder":
      return new THREE.CylinderGeometry(r, r, h, 32);
    case "cone":
      return new THREE.ConeGeometry(r, h, 32);
    case "torus":
      return new THREE.TorusGeometry(r, tube, 20, 36);
    case "plane":
      return new THREE.PlaneGeometry(w, h);
    default:
      // "group" has no geometry; any unexpected type degrades to a marker box.
      return null;
  }
}

function applyTransform(obj, node) {
  const pos = node.position || [0, 0, 0];
  const rot = node.rotation || [0, 0, 0];
  const scl = node.scale || [1, 1, 1];
  obj.position.set(pos[0] || 0, pos[1] || 0, pos[2] || 0);
  obj.rotation.set(rot[0] || 0, rot[1] || 0, rot[2] || 0);
  obj.scale.set(
    scl[0] === 0 ? 1 : scl[0] || 1,
    scl[1] === 0 ? 1 : scl[1] || 1,
    scl[2] === 0 ? 1 : scl[2] || 1
  );
}

// Recursively build a node (and its children) into a THREE.Object3D.
function buildNode(node) {
  const geometry = node.type === "group" ? null : buildGeometry(node.type, node.params);
  let obj;
  if (geometry) {
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(node.color || "#cccccc"),
      metalness: typeof node.metalness === "number" ? node.metalness : 0.0,
      roughness: typeof node.roughness === "number" ? node.roughness : 0.8,
      side: node.type === "plane" ? THREE.DoubleSide : THREE.FrontSide,
    });
    obj = new THREE.Mesh(geometry, material);
  } else {
    obj = new THREE.Group();
  }
  obj.name = node.id || node.type || "node";
  obj.userData.nodeId = node.id;
  applyTransform(obj, node);

  const children = Array.isArray(node.children) ? node.children : [];
  for (const child of children) {
    obj.add(buildNode(child));
  }
  return obj;
}

function disposeGroup(group) {
  group.traverse((o) => {
    if (o.isMesh) {
      o.geometry?.dispose();
      if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
      else o.material?.dispose();
    }
  });
}

// Replace the live model with one built from a validated Scene object.
function renderScene(sceneData) {
  clearSelection();
  scene.remove(modelGroup);
  disposeGroup(modelGroup);
  modelGroup = new THREE.Group();

  scene.background = new THREE.Color(sceneData.background || DEFAULT_BG);
  const nodes = Array.isArray(sceneData.nodes) ? sceneData.nodes : [];
  for (const node of nodes) {
    try {
      modelGroup.add(buildNode(node));
    } catch (err) {
      console.warn("skipping malformed node", node, err);
    }
  }
  scene.add(modelGroup);
  frameModel();
  applyWireframe(wireframeOn);
  setHud("status", `built “${sceneData.name || "scene"}” (${nodes.length} nodes)`);
}

// Frame the camera/orbit target on the new model's bounding box.
function frameModel() {
  const box = new THREE.Box3().setFromObject(modelGroup);
  if (box.isEmpty()) {
    controls.target.set(0, 0.5, 0);
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  controls.update();
}

// ── Mesh-from-URL (hi-fi path) ───────────────────────────────────────────────
// The backend may return {kind:"mesh", url, format} from the external hi-fi
// provider. We only ship GLTF/GLB loading here (the common case); other formats
// surface a friendly note. This never executes server code — it loads geometry.
async function renderMeshUrl(payload) {
  const url = payload.url;
  const format = (payload.format || "").toLowerCase();
  if (!url) {
    toast("Hi-fi response had no mesh URL; nothing to show.", "warn");
    return;
  }
  if (format && !["glb", "gltf"].includes(format)) {
    toast(`Hi-fi returned a ${format} mesh; only glb/gltf preview is supported.`, "warn");
    return;
  }
  try {
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    clearSelection();
    scene.remove(modelGroup);
    disposeGroup(modelGroup);
    modelGroup = new THREE.Group();
    modelGroup.add(gltf.scene);
    scene.add(modelGroup);
    frameModel();
    applyWireframe(wireframeOn);
    setHud("status", "loaded hi-fi mesh");
  } catch (err) {
    console.error(err);
    toast("Couldn't load the hi-fi mesh; try Fast quality.", "error");
  }
}

// ── Backend: generate ─────────────────────────────────────────────────────────
let currentQuality = "fast";
let generating = false;

async function generate(description) {
  const text = (description || "").trim();
  if (!text) {
    toast("Describe something first, Boss.", "warn");
    return;
  }
  if (generating) return;
  generating = true;
  setHud("status", `generating: ${text}`);
  if (promptInput && document.activeElement !== promptInput) promptInput.value = text;

  try {
    const res = await fetch("/studio/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ description: text, quality: currentQuality }),
    });
    if (res.status === 404) {
      setConn("bad", "studio disabled");
      toast("Studio is off. Set FRIDAY_ENABLE_STUDIO=true and restart.", "error");
      return;
    }
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail || body.error || detail;
      } catch (_) {
        /* non-JSON error body; keep the status text */
      }
      setConn("bad", "error");
      toast(`Generation failed: ${detail}`, "error");
      return;
    }
    setConn("ok", "connected");
    const data = await res.json();
    if (data.kind === "mesh") {
      await renderMeshUrl(data);
    } else if (data.kind === "scene" && data.scene) {
      renderScene(data.scene);
    } else if (data.scene) {
      // Tolerate a bare {scene:{...}} shape.
      renderScene(data.scene);
    } else {
      toast("Unexpected response shape from the studio API.", "warn");
    }
  } catch (err) {
    console.error(err);
    setConn("bad", "offline");
    toast("Can't reach the FRIDAY API. Is it running on :8000?", "error");
  } finally {
    generating = false;
  }
}

// ── Local operations (voice + buttons) ───────────────────────────────────────
let wireframeOn = false;

function applyWireframe(on) {
  modelGroup.traverse((o) => {
    if (o.isMesh && o.material) {
      if (Array.isArray(o.material)) o.material.forEach((m) => (m.wireframe = on));
      else o.material.wireframe = on;
    }
  });
}

function setWireframe(on) {
  wireframeOn = on;
  applyWireframe(on);
  if (wireToggleBtn) wireToggleBtn.setAttribute("aria-pressed", String(on));
  setHud("status", on ? "wireframe on" : "wireframe off");
}

function resetView() {
  camera.position.set(4, 3, 6);
  frameModel();
  setHud("status", "view reset");
}

function zoom(factor) {
  // Dolly the camera toward/away from the orbit target.
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
  dir.multiplyScalar(factor);
  camera.position.copy(controls.target).add(dir);
  controls.update();
}

function rotateModel(dx, dy) {
  modelGroup.rotation.y += dx;
  modelGroup.rotation.x += dy;
}

function scaleModel(factor) {
  const s = THREE.MathUtils.clamp(modelGroup.scale.x * factor, 0.1, 12);
  modelGroup.scale.setScalar(s);
}

const NAMED_COLORS = {
  red: "#ef4444", green: "#22c55e", blue: "#3b82f6", yellow: "#eab308",
  orange: "#f97316", purple: "#a855f7", pink: "#ec4899", white: "#f5f5f5",
  black: "#222228", gray: "#9ca3af", grey: "#9ca3af", cyan: "#06b6d4",
  gold: "#d4af37", silver: "#c0c0c0", magenta: "#d946ef", teal: "#14b8a6",
};

function colorModel(name) {
  const hex = NAMED_COLORS[(name || "").toLowerCase()] || name;
  let color;
  try {
    color = new THREE.Color(hex);
  } catch (_) {
    toast(`Don't know the color "${name}".`, "warn");
    return;
  }
  const target = selected || modelGroup;
  target.traverse((o) => {
    if (o.isMesh && o.material && o.material.color) o.material.color.copy(color);
  });
  setHud("status", `colored ${selected ? "selection" : "model"} ${name}`);
}

// ── Selection / highlight ────────────────────────────────────────────────────
function clearSelection() {
  if (selected && selectedOriginalEmissive && selected.material?.emissive) {
    selected.material.emissive.copy(selectedOriginalEmissive);
  }
  selected = null;
  selectedOriginalEmissive = null;
}

function highlightMesh(mesh) {
  if (!mesh || mesh === selected) return;
  clearSelection();
  if (mesh.material && mesh.material.emissive) {
    selected = mesh;
    selectedOriginalEmissive = mesh.material.emissive.clone();
    mesh.material.emissive.set(0x335577);
  }
  setHud("status", `selected ${mesh.name}`);
}

// Highlight the mesh nearest a normalized screen point (0..1, MediaPipe space).
function highlightNearestToScreen(nx, ny) {
  const meshes = [];
  modelGroup.traverse((o) => o.isMesh && meshes.push(o));
  if (!meshes.length) return;
  const target = new THREE.Vector2(nx * 2 - 1, -(ny * 2 - 1));
  let best = null;
  let bestDist = Infinity;
  const v = new THREE.Vector3();
  for (const m of meshes) {
    m.getWorldPosition(v).project(camera);
    const d = Math.hypot(v.x - target.x, v.y - target.y);
    if (d < bestDist) {
      bestDist = d;
      best = m;
    }
  }
  if (best) highlightMesh(best);
}

// ── Export / download ────────────────────────────────────────────────────────
function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function exportScene(format) {
  if (!modelGroup.children.length) {
    toast("Nothing to export yet — generate a model first.", "warn");
    return;
  }
  const stamp = Date.now();
  try {
    if (format === "glb") {
      new GLTFExporter().parse(
        modelGroup,
        (result) => {
          const blob = new Blob([result], { type: "model/gltf-binary" });
          triggerDownload(blob, `friday-model-${stamp}.glb`);
          setHud("status", "exported GLB");
        },
        (err) => {
          console.error(err);
          toast("GLB export failed.", "error");
        },
        { binary: true }
      );
    } else if (format === "stl") {
      const data = new STLExporter().parse(modelGroup, { binary: false });
      triggerDownload(new Blob([data], { type: "model/stl" }), `friday-model-${stamp}.stl`);
      setHud("status", "exported STL");
    } else if (format === "obj") {
      const data = new OBJExporter().parse(modelGroup);
      triggerDownload(new Blob([data], { type: "model/obj" }), `friday-model-${stamp}.obj`);
      setHud("status", "exported OBJ");
    }
  } catch (err) {
    console.error(err);
    toast(`Export to ${format.toUpperCase()} failed.`, "error");
  }
}

// ── Voice (Web Speech API) ───────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let listening = false;

function parseVoiceCommand(raw) {
  const t = raw.toLowerCase().trim();
  const makeMatch = t.match(/^(?:make|create|build|generate|draw)\s+(.*)$/);
  if (makeMatch && makeMatch[1]) {
    return { op: "generate", arg: makeMatch[1] };
  }
  if (/\breset\b/.test(t)) return { op: "reset" };
  if (/\bwireframe\b|\bwire frame\b/.test(t)) return { op: "wireframe" };
  if (/\bzoom\s*in\b/.test(t)) return { op: "zoom", arg: "in" };
  if (/\bzoom\s*out\b/.test(t)) return { op: "zoom", arg: "out" };
  if (/\brotate\b|\bspin\b/.test(t)) return { op: "rotate" };
  const colorMatch = t.match(/\bcolou?r\s+(?:it\s+)?([a-z]+)\b/);
  if (colorMatch) return { op: "color", arg: colorMatch[1] };
  const dlMatch = t.match(/\b(?:download|export|save)\s+(?:as\s+)?(glb|stl|obj)\b/);
  if (dlMatch) return { op: "download", arg: dlMatch[1] };
  return { op: "unknown" };
}

function runVoiceCommand(cmd) {
  switch (cmd.op) {
    case "generate":
      generate(cmd.arg);
      break;
    case "reset":
      resetView();
      break;
    case "wireframe":
      setWireframe(!wireframeOn);
      break;
    case "zoom":
      zoom(cmd.arg === "in" ? 0.8 : 1.25);
      setHud("status", `zoom ${cmd.arg}`);
      break;
    case "rotate":
      rotateModel(Math.PI / 6, 0);
      setHud("status", "rotated");
      break;
    case "color":
      colorModel(cmd.arg);
      break;
    case "download":
      exportScene(cmd.arg);
      break;
    default:
      toast(`Didn't catch a command in “${cmd.raw || ""}”.`, "warn");
  }
}

function initVoice() {
  if (!SpeechRecognition) {
    if (micBtn) micBtn.disabled = true;
    setHud("voice", "unsupported");
    return false;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const result = event.results[event.results.length - 1];
    if (!result || !result.isFinal) return;
    const transcript = result[0].transcript.trim();
    setHud("heard", transcript);
    const cmd = parseVoiceCommand(transcript);
    cmd.raw = transcript;
    runVoiceCommand(cmd);
  };
  recognition.onerror = (event) => {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      toast("Microphone permission denied; voice is off.", "warn");
      stopListening();
    } else if (event.error === "no-speech") {
      // benign; keep listening
    } else {
      console.warn("speech error", event.error);
    }
  };
  recognition.onend = () => {
    // Chrome stops after a pause; restart while the user wants it on.
    if (listening) {
      try {
        recognition.start();
      } catch (_) {
        /* already starting */
      }
    }
  };
  return true;
}

function startListening() {
  if (!recognition && !initVoice()) {
    toast("Voice recognition isn't supported in this browser.", "warn");
    return;
  }
  try {
    recognition.start();
    listening = true;
    setHud("voice", "listening");
    if (micBtn) micBtn.setAttribute("aria-pressed", "true");
  } catch (_) {
    /* start() throws if already started; treat as listening */
    listening = true;
  }
}

function stopListening() {
  listening = false;
  if (recognition) {
    try {
      recognition.stop();
    } catch (_) {
      /* ignore */
    }
  }
  setHud("voice", "off");
  if (micBtn) micBtn.setAttribute("aria-pressed", "false");
}

// ── Hand-tracking (MediaPipe Hands) ──────────────────────────────────────────
let hands = null;
let mpCamera = null;
let handTrackingOn = false;
const overlayCtx = overlayEl ? overlayEl.getContext("2d") : null;

// Gesture smoothing/history.
const gestureState = {
  lastPalm: null, // {x, y} of an open hand for rotate deltas
  lastPinch: null, // thumb-index distance for zoom deltas
  lastSpread: null, // two-hand distance for scale deltas
  highlightCooldown: 0,
};

function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Heuristic: a finger is "extended" if its tip is farther from the wrist than its
// PIP joint (works for the upright-hand case MediaPipe normalizes to).
function fingerExtended(lm, tip, pip, wrist) {
  return dist2(lm[tip], lm[wrist]) > dist2(lm[pip], lm[wrist]);
}

function classifyHand(lm) {
  const wrist = 0;
  const indexUp = fingerExtended(lm, 8, 6, wrist);
  const middleUp = fingerExtended(lm, 12, 10, wrist);
  const ringUp = fingerExtended(lm, 16, 14, wrist);
  const pinkyUp = fingerExtended(lm, 20, 18, wrist);
  const pinch = dist2(lm[4], lm[8]); // thumb tip ↔ index tip
  const extendedCount = [indexUp, middleUp, ringUp, pinkyUp].filter(Boolean).length;
  const isPinch = pinch < 0.06;
  const isPoint = indexUp && !middleUp && !ringUp && !pinkyUp;
  const isOpen = extendedCount >= 3;
  return { pinch, isPinch, isPoint, isOpen, palm: lm[9], index: lm[8] };
}

function drawLandmarks(results) {
  if (!overlayCtx || !overlayEl) return;
  overlayEl.width = overlayEl.clientWidth || 160;
  overlayEl.height = overlayEl.clientHeight || 120;
  overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
  const lmList = results.multiHandLandmarks || [];
  const drawConnectors = window.drawConnectors;
  const drawLandmarksFn = window.drawLandmarks;
  const HAND_CONNECTIONS = window.HAND_CONNECTIONS;
  for (const lm of lmList) {
    if (drawConnectors && HAND_CONNECTIONS) {
      drawConnectors(overlayCtx, lm, HAND_CONNECTIONS, { color: "#5b8cff", lineWidth: 2 });
    }
    if (drawLandmarksFn) {
      drawLandmarksFn(overlayCtx, lm, { color: "#34d399", radius: 2 });
    }
  }
}

function onHandResults(results) {
  drawLandmarks(results);
  const hands_ = results.multiHandLandmarks || [];

  if (hands_.length === 0) {
    gestureState.lastPalm = null;
    gestureState.lastPinch = null;
    gestureState.lastSpread = null;
    setHud("gesture", "no hands");
    return;
  }

  // Two-hand spread → scale.
  if (hands_.length >= 2) {
    const c0 = hands_[0][9];
    const c1 = hands_[1][9];
    const spread = dist2(c0, c1);
    if (gestureState.lastSpread != null) {
      const delta = spread - gestureState.lastSpread;
      if (Math.abs(delta) > 0.005) scaleModel(1 + delta * 1.5);
    }
    gestureState.lastSpread = spread;
    gestureState.lastPalm = null;
    gestureState.lastPinch = null;
    setHud("gesture", "two-hand · scale");
    return;
  }
  gestureState.lastSpread = null;

  const hand = classifyHand(hands_[0]);

  if (hand.isPinch) {
    if (gestureState.lastPinch != null) {
      const delta = hand.pinch - gestureState.lastPinch;
      // Closing the pinch (delta<0) zooms in.
      if (Math.abs(delta) > 0.003) zoom(1 + delta * 4);
    }
    gestureState.lastPinch = hand.pinch;
    gestureState.lastPalm = null;
    setHud("gesture", "pinch · zoom");
    return;
  }
  gestureState.lastPinch = null;

  if (hand.isPoint) {
    setHud("gesture", "point · select");
    if (gestureState.highlightCooldown <= 0) {
      // MediaPipe x is mirrored relative to the user; flip to screen space.
      highlightNearestToScreen(1 - hand.index.x, hand.index.y);
      gestureState.highlightCooldown = 8;
    } else {
      gestureState.highlightCooldown -= 1;
    }
    gestureState.lastPalm = null;
    return;
  }

  if (hand.isOpen) {
    const palm = hand.palm;
    if (gestureState.lastPalm) {
      const dx = palm.x - gestureState.lastPalm.x;
      const dy = palm.y - gestureState.lastPalm.y;
      // Mirror x so moving your hand right rotates the model right.
      rotateModel(-dx * 4, dy * 4);
    }
    gestureState.lastPalm = { x: palm.x, y: palm.y };
    setHud("gesture", "open · rotate");
    return;
  }

  gestureState.lastPalm = null;
  setHud("gesture", "tracking");
}

async function startHandTracking() {
  if (handTrackingOn) return;
  if (typeof window.Hands === "undefined" || typeof window.Camera === "undefined") {
    toast("Hand-tracking library failed to load (offline?). Gestures unavailable.", "warn");
    setHud("gesture", "unavailable");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("This browser has no camera access. Gestures unavailable.", "warn");
    setHud("gesture", "no camera");
    return;
  }
  try {
    hands = new window.Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    hands.onResults(onHandResults);

    mpCamera = new window.Camera(webcamEl, {
      onFrame: async () => {
        await hands.send({ image: webcamEl });
      },
      width: 320,
      height: 240,
    });
    await mpCamera.start();

    handTrackingOn = true;
    webcamEl.classList.add("is-live");
    overlayEl.classList.add("is-live");
    if (camToggleBtn) camToggleBtn.setAttribute("aria-pressed", "true");
    setHud("gesture", "ready");
    toast("Hand control on. Pinch=zoom · open hand=rotate · two hands=scale.", "info");
  } catch (err) {
    console.error(err);
    handTrackingOn = false;
    setHud("gesture", "denied");
    toast("Camera permission denied; gestures are off.", "warn");
  }
}

function stopHandTracking() {
  handTrackingOn = false;
  if (mpCamera) {
    try {
      mpCamera.stop();
    } catch (_) {
      /* ignore */
    }
  }
  if (webcamEl) webcamEl.classList.remove("is-live");
  if (overlayEl) overlayEl.classList.remove("is-live");
  if (overlayCtx && overlayEl) overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
  if (camToggleBtn) camToggleBtn.setAttribute("aria-pressed", "false");
  setHud("gesture", "off");
}

// ── Resize + render loop ─────────────────────────────────────────────────────
function resize() {
  const w = viewport.clientWidth || window.innerWidth;
  const h = viewport.clientHeight || window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ── Health check ─────────────────────────────────────────────────────────────
async function pingApi() {
  try {
    const res = await fetch("/health", { method: "GET" });
    if (res.ok) setConn("ok", "connected");
    else setConn("bad", "api error");
  } catch (_) {
    setConn("bad", "offline");
  }
}

// ── Wire up the UI ───────────────────────────────────────────────────────────
function wireEvents() {
  if (promptForm) {
    promptForm.addEventListener("submit", (e) => {
      e.preventDefault();
      generate(promptInput ? promptInput.value : "");
    });
  }

  if (micBtn) {
    micBtn.addEventListener("click", () => {
      if (listening) stopListening();
      else startListening();
    });
  }

  if (camToggleBtn) {
    camToggleBtn.addEventListener("click", () => {
      if (handTrackingOn) stopHandTracking();
      else startHandTracking();
    });
  }

  if (resetViewBtn) resetViewBtn.addEventListener("click", resetView);
  if (wireToggleBtn) wireToggleBtn.addEventListener("click", () => setWireframe(!wireframeOn));

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      currentQuality = btn.dataset.quality || "fast";
      setHud("status", `quality: ${currentQuality}`);
    });
  });

  document.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", () => exportScene(btn.dataset.export));
  });

  // Click-to-select on the canvas (mouse fallback for the point gesture).
  renderer.domElement.addEventListener("click", (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width;
    const ny = (event.clientY - rect.top) / rect.height;
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(nx * 2 - 1, -(ny * 2 - 1)), camera);
    const hits = raycaster.intersectObject(modelGroup, true);
    const meshHit = hits.find((h) => h.object.isMesh);
    if (meshHit) highlightMesh(meshHit.object);
    else clearSelection();
  });
}

// ── A friendly starter scene so the canvas isn't empty on first load ─────────
const STARTER_SCENE = {
  name: "FRIDAY welcome",
  background: DEFAULT_BG,
  nodes: [
    {
      id: "body",
      type: "box",
      params: { w: 1.2, h: 1.6, d: 0.8 },
      position: [0, 0.9, 0],
      color: "#5b8cff",
      metalness: 0.3,
      roughness: 0.5,
      children: [
        {
          id: "head",
          type: "sphere",
          params: { r: 0.5 },
          position: [0, 1.2, 0],
          color: "#e7e7ef",
          metalness: 0.1,
          roughness: 0.6,
        },
      ],
    },
    {
      id: "base",
      type: "cylinder",
      params: { r: 0.9, h: 0.2 },
      position: [0, 0.1, 0],
      color: "#34d399",
      metalness: 0.4,
      roughness: 0.4,
    },
  ],
};

function init() {
  resize();
  wireEvents();
  renderScene(STARTER_SCENE);
  animate();
  pingApi();
  if (!SpeechRecognition && micBtn) {
    micBtn.disabled = true;
    micBtn.title = "Voice not supported in this browser";
    setHud("voice", "unsupported");
  }
  setHud("status", "ready — describe a model, or say “make a red sphere”");
}

init();
