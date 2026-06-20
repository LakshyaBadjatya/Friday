/* © Lakshya Badjatya — Author */
/*
  FRIDAY HUD — hud-fx.js : the "FRIDAY OS" maximal effects layer.

  PURELY ADDITIVE. This script never touches hud.js's internals; it only reads
  the DOM that index.html ships and paints decorative effects on top:

    • matrix-rain backdrop (#matrix), theme-tinted, gated like the particle field
    • live rail telemetry (#tele-*) — uplink latency from /health, drifting load
    • holographic "materialise" reveal on freshly rendered cards/turns
    • a periodic glitch on the FRIDAY wordmark
    • an opt-in WebAudio interface-sound engine (off until armed)
    • an audio-reactive reactor while the wake mic is listening
    • calm-mode + sound toggles, both persisted to localStorage

  Everything self-disables under prefers-reduced-motion, calm mode, small
  screens, or a hidden tab. If anything throws, it fails silent — the cockpit
  keeps working exactly as before.
*/
(function () {
  "use strict";

  var doc = document;
  var root = doc.documentElement;
  function $(id) { return doc.getElementById(id); }
  function reduced() {
    try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }
    catch (e) { return false; }
  }
  function lsGet(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { window.localStorage.setItem(k, v); } catch (e) {} }

  var KEY_CALM = "friday-fx-calm";
  var KEY_SOUND = "friday-fx-sound";
  var calm = lsGet(KEY_CALM) === "1";
  var soundArmed = lsGet(KEY_SOUND) === "1";

  /* cached, theme-aware accent ------------------------------------------- */
  var accentRgb = "79, 227, 255";
  var monoFont = '"SF Mono", ui-monospace, monospace';
  function refreshTokens() {
    try {
      var cs = getComputedStyle(root);
      var r = cs.getPropertyValue("--cyan-rgb").trim();
      if (r) accentRgb = r;
      var m = cs.getPropertyValue("--mono").trim();
      if (m) monoFont = m;
    } catch (e) {}
  }

  /* ====================================================== MATRIX RAIN ==== */
  var mCanvas, mctx, mCols = [], mFrame = 0, mRaf = null;
  var MFONT = 14;
  var GLYPHS = "アカサタナハマヤラ0123456789ABCDEF<>/\\|=+*".split("");

  function matrixOK() {
    return !calm && !reduced() && window.innerWidth >= 900 && !doc.hidden;
  }
  function matrixResize() {
    if (!mCanvas) return;
    mCanvas.width = window.innerWidth;
    mCanvas.height = window.innerHeight;
    var n = Math.floor(mCanvas.width / MFONT);
    mCols = [];
    for (var i = 0; i < n; i++) mCols[i] = Math.random() * -60;
  }
  function matrixStep() {
    if (!matrixOK()) { mRaf = null; return; }
    mRaf = window.requestAnimationFrame(matrixStep);
    if ((mFrame++ & 1) === 0) return; /* throttle to ~30fps */
    mctx.fillStyle = "rgba(3, 6, 13, 0.085)";
    mctx.fillRect(0, 0, mCanvas.width, mCanvas.height);
    mctx.font = MFONT + "px " + monoFont;
    for (var i = 0; i < mCols.length; i++) {
      var y = mCols[i];
      var x = i * MFONT;
      var g = GLYPHS[(Math.random() * GLYPHS.length) | 0];
      /* bright leading glyph, faint trail */
      mctx.fillStyle = "rgba(" + accentRgb + ", 0.85)";
      mctx.fillText(g, x, y * MFONT);
      mctx.fillStyle = "rgba(" + accentRgb + ", 0.18)";
      mctx.fillText(GLYPHS[(Math.random() * GLYPHS.length) | 0], x, (y - 1) * MFONT);
      if (y * MFONT > mCanvas.height && Math.random() > 0.975) mCols[i] = 0;
      else mCols[i] = y + 1;
    }
  }
  function matrixStart() {
    if (!mCanvas || mRaf || !matrixOK()) return;
    matrixResize();
    mRaf = window.requestAnimationFrame(matrixStep);
  }
  function matrixStop() {
    if (mRaf) { window.cancelAnimationFrame(mRaf); mRaf = null; }
    if (mctx) mctx.clearRect(0, 0, mCanvas.width, mCanvas.height);
  }

  /* ======================================================= TELEMETRY ==== */
  // Mirror hud.js: default to the live Render backend so the uplink ping probes
  // the one real backend even when this page is opened from localhost.
  var apiBase = (window.FRIDAY_API_BASE || "https://friday-backend-oj8h.onrender.com").replace(/\/$/, "");
  var telePhase = 0;
  function setBar(id, pct) { var n = $(id); if (n) n.style.width = Math.max(4, Math.min(100, pct)) + "%"; }
  function setVal(id, v) { var n = $(id); if (n) n.textContent = v; }
  function hex4() { return "0x" + (((Math.random() * 0xffff) | 0).toString(16).toUpperCase().padStart(4, "0")); }

  async function pingUplink() {
    var t0 = (window.performance && performance.now) ? performance.now() : Date.now();
    try {
      var resp = await fetch(apiBase + "/health", { method: "GET", cache: "no-store" });
      var ms = Math.round(((window.performance && performance.now) ? performance.now() : Date.now()) - t0);
      setVal("tele-uplink", resp.ok ? ms + "ms" : "DOWN");
      setBar("tele-uplink-bar", resp.ok ? Math.max(8, 100 - ms / 3) : 6);
    } catch (e) {
      setVal("tele-uplink", "----");
      setBar("tele-uplink-bar", 6);
    }
  }
  function teleTick() {
    telePhase += 0.6;
    var neural = 72 + Math.sin(telePhase) * 16 + Math.random() * 6;
    var core = 58 + Math.cos(telePhase * 0.7) * 22 + Math.random() * 5;
    setBar("tele-neural-bar", neural);
    setVal("tele-neural", Math.round(neural) + "%");
    setBar("tele-core-bar", core);
    setVal("tele-core", Math.round(core) + "%");
    setVal("tele-hex", hex4() + " · " + hex4() + " · " + hex4());
  }

  /* ================================================ HOLOGRAPHIC REVEAL === */
  function reveal(node) {
    if (reduced() || node.nodeType !== 1) return;
    node.classList.add("fx-reveal");
    window.setTimeout(function () { node.classList.remove("fx-reveal"); }, 420);
  }
  function watchReveal() {
    if (typeof MutationObserver === "undefined") return;
    var ids = ["chat-log", "agents-grid", "arena-results", "dossier-cards",
      "rag-sources", "journal-list", "audit-list", "flow-list", "brain-list",
      "by-mode", "graph-entities"];
    var obs = new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        var added = muts[i].addedNodes;
        for (var j = 0; j < added.length; j++) reveal(added[j]);
      }
    });
    for (var k = 0; k < ids.length; k++) {
      var n = $(ids[k]);
      if (n) obs.observe(n, { childList: true });
    }
  }

  /* ====================================================== WORDMARK GLITCH = */
  var brand;
  function glitch() {
    if (!brand || calm || reduced() || doc.hidden) return;
    brand.classList.add("is-glitch");
    window.setTimeout(function () { brand.classList.remove("is-glitch"); }, 480);
  }

  /* ========================================================= SOUND ENGINE = */
  var actx = null;
  function audio() {
    if (!soundArmed) return null;
    try {
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === "suspended") actx.resume();
      return actx;
    } catch (e) { return null; }
  }
  function blip(freq, dur, type, gain) {
    var a = audio();
    if (!a) return;
    try {
      var osc = a.createOscillator();
      var g = a.createGain();
      osc.type = type || "sine";
      osc.frequency.value = freq;
      g.gain.value = 0.0001;
      osc.connect(g); g.connect(a.destination);
      var t = a.currentTime;
      g.gain.exponentialRampToValueAtTime(gain || 0.06, t + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0001, t + (dur || 0.12));
      osc.start(t); osc.stop(t + (dur || 0.12) + 0.02);
    } catch (e) {}
  }
  var SND = {
    send: function () { blip(660, 0.1, "triangle", 0.05); },
    tick: function () { blip(420, 0.05, "square", 0.03); },
    wake: function () { blip(880, 0.16, "sine", 0.06); window.setTimeout(function () { blip(1320, 0.12, "sine", 0.05); }, 90); },
    boot: function () { blip(180, 0.5, "sawtooth", 0.04); window.setTimeout(function () { blip(720, 0.3, "sine", 0.05); }, 220); }
  };

  /* ============================================ AUDIO-REACTIVE REACTOR === */
  var mic = { stream: null, raf: null };
  async function reactorOn() {
    if (mic.stream || reduced()) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { reactorSynthetic(); return; }
    try {
      var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mic.stream = stream;
      var a = new (window.AudioContext || window.webkitAudioContext)();
      var src = a.createMediaStreamSource(stream);
      var an = a.createAnalyser();
      an.fftSize = 256;
      src.connect(an);
      var buf = new Uint8Array(an.frequencyBinCount);
      var loop = function () {
        if (!mic.stream) return;
        mic.raf = window.requestAnimationFrame(loop);
        an.getByteFrequencyData(buf);
        var sum = 0;
        for (var i = 0; i < buf.length; i++) sum += buf[i];
        var level = Math.min(1, (sum / buf.length) / 90);
        root.style.setProperty("--level", level.toFixed(3));
      };
      loop();
    } catch (e) { reactorSynthetic(); }
  }
  function reactorSynthetic() {
    /* no mic access — fake a gentle reactive pulse so the reactor still breathes */
    var p = 0;
    var loop = function () {
      if (!micListening) { root.style.setProperty("--level", "0"); mic.raf = null; return; }
      mic.raf = window.requestAnimationFrame(loop);
      p += 0.18;
      root.style.setProperty("--level", (0.22 + Math.abs(Math.sin(p)) * 0.5).toFixed(3));
    };
    loop();
  }
  function reactorOff() {
    if (mic.raf) { window.cancelAnimationFrame(mic.raf); mic.raf = null; }
    if (mic.stream) { try { mic.stream.getTracks().forEach(function (t) { t.stop(); }); } catch (e) {} mic.stream = null; }
    root.style.setProperty("--level", "0");
  }
  var micListening = false;
  function watchMic() {
    var btn = $("btn-mic");
    if (!btn || typeof MutationObserver === "undefined") return;
    var obs = new MutationObserver(function () {
      var on = btn.classList.contains("is-listening");
      if (on === micListening) return;
      micListening = on;
      if (on) { SND.wake(); if (soundArmed) reactorOn(); else reactorSynthetic(); }
      else { reactorOff(); }
    });
    obs.observe(btn, { attributes: true, attributeFilter: ["class"] });
  }

  /* ============================================ TOGGLES + EVENT WIRING === */
  function syncToggle(btn, on) { if (btn) btn.setAttribute("aria-pressed", on ? "true" : "false"); }
  function applyCalm() {
    root.classList.toggle("calm", calm);
    if (calm) matrixStop(); else matrixStart();
  }
  function wireToggles() {
    var calmBtn = $("fx-calm");
    var soundBtn = $("fx-sound");
    syncToggle(calmBtn, calm);
    syncToggle(soundBtn, soundArmed);
    if (calmBtn) calmBtn.addEventListener("click", function () {
      calm = !calm; lsSet(KEY_CALM, calm ? "1" : "0"); syncToggle(calmBtn, calm); applyCalm();
    });
    if (soundBtn) soundBtn.addEventListener("click", function () {
      soundArmed = !soundArmed; lsSet(KEY_SOUND, soundArmed ? "1" : "0"); syncToggle(soundBtn, soundArmed);
      if (soundArmed) { audio(); SND.tick(); }
    });
  }
  function wireSounds() {
    var send = $("composer-send");
    if (send) send.addEventListener("click", function () { SND.send(); });
    var form = $("composer");
    if (form) form.addEventListener("submit", function () { SND.send(); });
    var rail = doc.querySelector(".rail-nav");
    if (rail) rail.addEventListener("click", function (e) {
      if (e.target && e.target.closest && e.target.closest(".rail-item")) SND.tick();
    });
  }

  /* ====================================================== THEME WATCH ==== */
  function watchTheme() {
    if (typeof MutationObserver === "undefined") return;
    var obs = new MutationObserver(function () { refreshTokens(); glitch(); });
    obs.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  }

  /* ============================================================= INIT ==== */
  function init() {
    refreshTokens();
    brand = doc.querySelector(".brand-word");

    mCanvas = $("matrix");
    if (mCanvas && mCanvas.getContext) {
      mctx = mCanvas.getContext("2d");
      window.addEventListener("resize", function () { if (matrixOK()) matrixResize(); });
    }

    applyCalm();          /* sets html.calm + starts/stops matrix */
    watchReveal();
    watchTheme();
    watchMic();
    wireToggles();
    wireSounds();

    /* telemetry: ping uplink slowly, drift the load bars quickly */
    pingUplink();
    window.setInterval(pingUplink, 6000);
    teleTick();
    window.setInterval(teleTick, 1400);

    /* periodic wordmark glitch + one welcome glitch + boot sound */
    window.setTimeout(function () { glitch(); SND.boot(); }, 2400);
    window.setInterval(glitch, 13000);

    /* pause/resume the heavy canvas with tab visibility */
    doc.addEventListener("visibilitychange", function () {
      if (doc.hidden) matrixStop(); else matrixStart();
    });
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", init);
  else init();
})();
