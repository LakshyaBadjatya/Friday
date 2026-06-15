/*
  FRIDAY HUD — cockpit controller (no-build, vanilla ES2022).

  Three jobs, all client-side and offline-safe:

  1. A drifting particle field rendered on a full-viewport <canvas> (pure 2D, no
     libraries) for the "glow" cockpit feel.
  2. An arc-reactor BOOT SEQUENCE: a short scripted ramp ("initializing reactor…"
     -> "online") that fades the overlay out, while a live /admin/state probe
     decides whether the backend is actually reachable.
  3. A command palette (Cmd/Ctrl-K) that drives the EXISTING same-origin
     endpoints — POST /chat (ask FRIDAY) and the read-only GET /admin/* surfaces
     (state, metrics, flags). Results are rendered as TEXT; no server output is
     ever eval()'d or injected as HTML.

  Everything degrades gracefully: if a fetch fails, the palette shows a friendly
  error and the connection pill flips to "offline" — the cockpit never crashes.
*/

(function () {
  "use strict";

  // --- tiny DOM helpers ------------------------------------------------------
  /** @param {string} id */
  function el(id) {
    return document.getElementById(id);
  }

  /** Append a timestamped line to the activity log (text only, never HTML). */
  function logLine(message, kind) {
    var log = el("log");
    if (!log) {
      return;
    }
    var row = document.createElement("div");
    row.className = "line" + (kind ? " " + kind : "");
    var time = document.createElement("span");
    time.className = "t";
    var now = new Date();
    time.textContent =
      "[" +
      String(now.getHours()).padStart(2, "0") +
      ":" +
      String(now.getMinutes()).padStart(2, "0") +
      ":" +
      String(now.getSeconds()).padStart(2, "0") +
      "] ";
    row.appendChild(time);
    row.appendChild(document.createTextNode(String(message)));
    log.insertBefore(row, log.firstChild);
    // Keep the log bounded so it never grows without limit.
    while (log.childNodes.length > 40) {
      log.removeChild(log.lastChild);
    }
  }

  // --- backend client (same-origin; no key baked in) -------------------------
  // The HUD is served same-origin with the API, so all requests are relative.
  // Auth (if the gateway requires it) is the browser/gateway's concern; the HUD
  // ships no credentials of its own.

  /** GET a JSON endpoint; resolves to parsed JSON or throws. */
  async function getJSON(path) {
    var resp = await fetch(path, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!resp.ok) {
      throw new Error("GET " + path + " -> HTTP " + resp.status);
    }
    return resp.json();
  }

  /** POST a JSON body to an endpoint; resolves to parsed JSON or throws. */
  async function postJSON(path, body) {
    var resp = await fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error("POST " + path + " -> HTTP " + resp.status);
    }
    return resp.json();
  }

  // A stable session id for this HUD tab's /chat turns.
  var SESSION_ID = "hud-" + Math.random().toString(36).slice(2, 10);

  // --- particle field --------------------------------------------------------
  function startParticles() {
    var canvas = el("particles");
    if (!canvas || !canvas.getContext) {
      return;
    }
    var ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    var particles = [];

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    var COUNT = 70;
    for (var i = 0; i < COUNT; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.8 + 0.4,
        a: Math.random() * 0.5 + 0.2,
      });
    }

    function frame() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (var i = 0; i < particles.length; i++) {
        var p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(65, 230, 255, " + p.a + ")";
        ctx.shadowColor = "rgba(65, 230, 255, 0.8)";
        ctx.shadowBlur = 8;
        ctx.fill();
      }
      ctx.shadowBlur = 0;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // --- connection pill -------------------------------------------------------
  function setOnline(online) {
    var pill = el("conn-status");
    var reactor = el("reactor-state");
    if (pill) {
      pill.classList.toggle("is-online", online);
      pill.classList.toggle("is-offline", !online);
      var label = pill.querySelector(".conn-label");
      if (label) {
        label.textContent = online ? "online" : "offline";
      }
    }
    if (reactor) {
      reactor.textContent = online ? "ONLINE" : "STANDBY";
    }
  }

  // --- telemetry refresh -----------------------------------------------------
  async function refreshTelemetry() {
    try {
      var metrics = await getJSON("/admin/metrics");
      var setText = function (id, v) {
        var node = el(id);
        if (node) {
          node.textContent = v == null ? "—" : String(v);
        }
      };
      setText("m-requests", metrics.requests);
      setText("m-tools", metrics.tool_calls);
      setText("m-errors", metrics.errors);
      setOnline(true);
    } catch (err) {
      setOnline(false);
      return;
    }
    try {
      var state = await getJSON("/admin/state");
      var sessions = state && state.sessions ? state.sessions.length : 0;
      var node = el("m-sessions");
      if (node) {
        node.textContent = String(sessions);
      }
    } catch (err) {
      /* state is best-effort; metrics already drove the pill. */
    }
  }

  // --- arc-reactor boot sequence ---------------------------------------------
  function runBootSequence() {
    var boot = el("boot");
    var text = el("boot-text");
    var bar = el("boot-bar");
    var steps = [
      "initializing reactor…",
      "spinning up containment rings…",
      "calibrating telemetry…",
      "linking command bus…",
      "reactor online.",
    ];
    var i = 0;

    function tick() {
      if (text) {
        text.textContent = steps[i];
      }
      if (bar) {
        bar.style.width = Math.round(((i + 1) / steps.length) * 100) + "%";
      }
      i++;
      if (i < steps.length) {
        window.setTimeout(tick, 480);
      } else {
        window.setTimeout(function () {
          if (boot) {
            boot.classList.add("is-done");
          }
          logLine("reactor online", "ok");
        }, 520);
      }
    }
    tick();
  }

  // --- command palette -------------------------------------------------------
  // Each command is a small async action over the existing endpoints. A "free"
  // typed query (not matching a command) is sent to FRIDAY via POST /chat.
  var COMMANDS = [
    {
      id: "ask",
      title: "Ask FRIDAY",
      hint: "POST /chat",
      glyph: "▸",
      run: async function (query) {
        var q = (query || "").trim();
        if (!q) {
          return "Type your question after the command, then Enter.";
        }
        var data = await postJSON("/chat", { session_id: SESSION_ID, text: q });
        logLine("chat: " + q, "ok");
        return String(data && data.text ? data.text : "(no reply)");
      },
    },
    {
      id: "metrics",
      title: "Show metrics",
      hint: "GET /admin/metrics",
      glyph: "◴",
      run: async function () {
        var m = await getJSON("/admin/metrics");
        return JSON.stringify(m, null, 2);
      },
    },
    {
      id: "state",
      title: "Show state",
      hint: "GET /admin/state",
      glyph: "◷",
      run: async function () {
        var s = await getJSON("/admin/state");
        var n = s && s.sessions ? s.sessions.length : 0;
        return n + " active session(s)\n" + JSON.stringify(s, null, 2);
      },
    },
    {
      id: "flags",
      title: "Show feature flags",
      hint: "GET /admin/flags",
      glyph: "⚑",
      run: async function () {
        var f = await getJSON("/admin/flags");
        return JSON.stringify(f.flags || f, null, 2);
      },
    },
  ];

  var paletteState = { open: false, active: 0, filtered: COMMANDS.slice() };

  function isFreeQuery(value) {
    // A leading "/" or empty value uses command matching; anything else that
    // doesn't match a command title is treated as a free question for /chat.
    return value && value.trim().length > 0;
  }

  function filterCommands(value) {
    var v = (value || "").trim().toLowerCase();
    if (!v) {
      return COMMANDS.slice();
    }
    var matches = COMMANDS.filter(function (c) {
      return (
        c.title.toLowerCase().indexOf(v) !== -1 ||
        c.id.toLowerCase().indexOf(v) !== -1
      );
    });
    // If nothing matches a command, surface "Ask FRIDAY" so the query goes to /chat.
    if (matches.length === 0) {
      return [COMMANDS[0]];
    }
    return matches;
  }

  function renderPaletteList() {
    var list = el("palette-list");
    if (!list) {
      return;
    }
    list.textContent = "";
    paletteState.filtered.forEach(function (cmd, idx) {
      var li = document.createElement("li");
      li.className = idx === paletteState.active ? "is-active" : "";
      var glyph = document.createElement("span");
      glyph.className = "cmd-glyph";
      glyph.textContent = cmd.glyph || "▸";
      var title = document.createElement("span");
      title.className = "cmd-title";
      title.textContent = cmd.title;
      var hint = document.createElement("span");
      hint.className = "cmd-hint";
      hint.textContent = cmd.hint || "";
      li.appendChild(glyph);
      li.appendChild(title);
      li.appendChild(hint);
      li.addEventListener("click", function () {
        paletteState.active = idx;
        runActiveCommand();
      });
      list.appendChild(li);
    });
  }

  function openPalette() {
    var overlay = el("palette-overlay");
    var input = el("palette-input");
    var result = el("palette-result");
    if (!overlay || !input) {
      return;
    }
    paletteState.open = true;
    paletteState.active = 0;
    paletteState.filtered = COMMANDS.slice();
    overlay.classList.add("is-open");
    if (result) {
      result.textContent = "";
      result.classList.remove("is-err");
    }
    input.value = "";
    renderPaletteList();
    window.setTimeout(function () {
      input.focus();
    }, 0);
  }

  function closePalette() {
    var overlay = el("palette-overlay");
    if (!overlay) {
      return;
    }
    paletteState.open = false;
    overlay.classList.remove("is-open");
  }

  function showResult(text, isErr) {
    var result = el("palette-result");
    if (!result) {
      return;
    }
    result.classList.toggle("is-err", !!isErr);
    result.textContent = String(text);
  }

  async function runActiveCommand() {
    var input = el("palette-input");
    var raw = input ? input.value : "";
    var cmd = paletteState.filtered[paletteState.active] || COMMANDS[0];
    showResult("…");
    try {
      // The free-query path passes the raw text to "Ask FRIDAY"; command
      // selections that take no argument ignore it.
      var arg = cmd.id === "ask" ? raw : "";
      var out = await cmd.run(arg);
      showResult(out, false);
    } catch (err) {
      logLine(String(err && err.message ? err.message : err), "err");
      showResult(
        "Command failed: " + (err && err.message ? err.message : err),
        true
      );
      setOnline(false);
    }
  }

  function wirePalette() {
    var input = el("palette-input");
    var overlay = el("palette-overlay");
    if (!input || !overlay) {
      return;
    }

    input.addEventListener("input", function () {
      paletteState.filtered = filterCommands(input.value);
      paletteState.active = 0;
      renderPaletteList();
    });

    input.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        paletteState.active =
          (paletteState.active + 1) % paletteState.filtered.length;
        renderPaletteList();
      } else if (ev.key === "ArrowUp") {
        ev.preventDefault();
        paletteState.active =
          (paletteState.active - 1 + paletteState.filtered.length) %
          paletteState.filtered.length;
        renderPaletteList();
      } else if (ev.key === "Enter") {
        ev.preventDefault();
        // A free question that matches no command still flows to /chat via the
        // surfaced "Ask FRIDAY" entry.
        if (
          isFreeQuery(input.value) &&
          paletteState.filtered.length === 1 &&
          paletteState.filtered[0].id === "ask"
        ) {
          paletteState.active = 0;
        }
        runActiveCommand();
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        closePalette();
      }
    });

    // Clicking the dimmed backdrop closes the palette.
    overlay.addEventListener("click", function (ev) {
      if (ev.target === overlay) {
        closePalette();
      }
    });
  }

  function wireGlobalKeys() {
    window.addEventListener("keydown", function (ev) {
      var isPaletteKey =
        (ev.metaKey || ev.ctrlKey) && (ev.key === "k" || ev.key === "K");
      if (isPaletteKey) {
        ev.preventDefault();
        if (paletteState.open) {
          closePalette();
        } else {
          openPalette();
        }
      }
    });
  }

  // --- bootstrap -------------------------------------------------------------
  function init() {
    startParticles();
    wirePalette();
    wireGlobalKeys();
    runBootSequence();
    logLine("HUD loaded — press Cmd/Ctrl-K");
    refreshTelemetry();
    // Periodic telemetry refresh; failures flip the pill but never throw.
    window.setInterval(refreshTelemetry, 5000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
