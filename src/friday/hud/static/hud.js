// © Lakshya Badjatya — Author
/*
  FRIDAY HUD — "viewing space" cockpit controller (no-build, vanilla ES2022).

  A live single-page cockpit over the EXISTING same-origin FRIDAY endpoints. No
  bundler, no framework, no CDN: every line is plain ES2022 the browser runs
  directly. The page NEVER eval()s server output — endpoint data is rendered as
  TEXT nodes only, so a surprising payload can paint text but can never execute.

  The shell is a slim left rail + one large main canvas. Exactly one view is
  visible at a time (Command, Arena, Agents, Memory, System); switching is instant
  and sets the URL hash for deep-linking + back-button. Heavy polling (metrics,
  traces, audit) runs ONLY while System is the active view; a slow /health
  heartbeat keeps the connection pill honest everywhere. Every optional endpoint
  degrades gracefully — a 404 quietly hides its panel instead of crashing.

  API BASE is configurable: ?api=<url> in the page URL (or window.FRIDAY_API_BASE)
  points every fetch at a remote backend; default is same-origin ("").
*/

(function () {
  "use strict";

  // --- tiny DOM helpers ------------------------------------------------------
  /** @param {string} id */
  function el(id) {
    return document.getElementById(id);
  }

  /** Create an element with an optional class and text (text is a TEXT node). */
  function mk(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text != null) {
      node.textContent = String(text);
    }
    return node;
  }

  /** Remove all children of a node (safe re-render without innerHTML). */
  function clear(node) {
    if (!node) {
      return;
    }
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  /** Short HH:MM:SS stamp for the current (or a given) time. */
  function stamp(date) {
    var d = date || new Date();
    return (
      String(d.getHours()).padStart(2, "0") +
      ":" +
      String(d.getMinutes()).padStart(2, "0") +
      ":" +
      String(d.getSeconds()).padStart(2, "0")
    );
  }

  /** Append a timestamped line to the activity log (text only, never HTML). */
  function logLine(message, kind) {
    var log = el("log");
    if (!log) {
      return;
    }
    var row = mk("div", "line" + (kind ? " " + kind : ""));
    row.appendChild(mk("span", "t", "[" + stamp() + "] "));
    row.appendChild(document.createTextNode(String(message)));
    log.insertBefore(row, log.firstChild);
    while (log.childNodes.length > 60) {
      log.removeChild(log.lastChild);
    }
  }

  /** Show a transient toast message (auto-hides). */
  var toastTimer = null;
  function toast(message) {
    var node = el("toast");
    if (!node) {
      return;
    }
    node.textContent = String(message);
    node.hidden = false;
    if (toastTimer) {
      window.clearTimeout(toastTimer);
    }
    toastTimer = window.setTimeout(function () {
      node.hidden = true;
    }, 3600);
  }

  // --- API base (configurable) ----------------------------------------------
  function resolveApiBase() {
    var base = "";
    try {
      var params = new URLSearchParams(window.location.search);
      base = params.get("api") || window.FRIDAY_API_BASE || "";
    } catch (err) {
      base = window.FRIDAY_API_BASE || "";
    }
    return String(base).replace(/\/+$/, "");
  }
  var API_BASE = resolveApiBase();

  /** Join the API base with an endpoint path. */
  function url(path) {
    return API_BASE + path;
  }

  // --- backend client (no key baked in) -------------------------------------
  /** GET a JSON endpoint; resolves to parsed JSON or throws (status on .status). */
  async function getJSON(path) {
    var resp = await fetch(url(path), {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!resp.ok) {
      var e = new Error("GET " + path + " -> HTTP " + resp.status);
      e.status = resp.status;
      throw e;
    }
    return resp.json();
  }

  /** POST a JSON body to an endpoint; resolves to parsed JSON or throws. */
  async function postJSON(path, body) {
    var resp = await fetch(url(path), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      var e = new Error("POST " + path + " -> HTTP " + resp.status);
      e.status = resp.status;
      throw e;
    }
    return resp.json();
  }

  // A stable session id for this HUD tab's /chat turns.
  var SESSION_ID = "hud-" + Math.random().toString(36).slice(2, 10);

  // Who the next free-form question is addressed to (a persona name or null).
  var addressTarget = null;

  // The active model id (POST /chat passes it so a turn uses the chosen brain).
  var activeModel = null;

  // --- prefers-reduced-motion ------------------------------------------------
  function reducedMotion() {
    try {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (err) {
      return false;
    }
  }

  // --- particle field (light; disabled on small screens / reduced motion) ----
  function startParticles() {
    var canvas = el("particles");
    if (!canvas || !canvas.getContext) {
      return;
    }
    if (reducedMotion() || window.innerWidth < 900) {
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

    var COUNT = 34;
    for (var i = 0; i < COUNT; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.22,
        r: Math.random() * 1.4 + 0.4,
        a: Math.random() * 0.35 + 0.12,
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
        ctx.fillStyle = "rgba(79, 227, 255, " + p.a + ")";
        ctx.fill();
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // --- connection pill -------------------------------------------------------
  function setOnline(online) {
    var pill = el("conn-status");
    if (pill) {
      pill.classList.toggle("is-online", online);
      pill.classList.toggle("is-offline", !online);
      var label = pill.querySelector(".conn-label");
      if (label) {
        label.textContent = online ? "online" : "offline";
      }
    }
  }

  /** Slow global heartbeat: a cheap /health ping drives the connection pill. */
  async function heartbeat() {
    try {
      var resp = await fetch(url("/health"), { method: "GET" });
      setOnline(resp.ok);
    } catch (err) {
      setOnline(false);
    }
  }

  // --- arc-reactor boot sequence (brief) ------------------------------------
  function runBootSequence() {
    var boot = el("boot");
    var text = el("boot-text");
    var bar = el("boot-bar");
    if (reducedMotion()) {
      if (boot) {
        boot.classList.add("is-done");
      }
      logLine("reactor online", "ok");
      return;
    }
    var steps = [
      "initializing reactor…",
      "mustering roster…",
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
        window.setTimeout(tick, 380);
      } else {
        window.setTimeout(function () {
          if (boot) {
            boot.classList.add("is-done");
          }
          logLine("reactor online", "ok");
        }, 420);
      }
    }
    tick();
  }

  // ==========================================================================
  // VIEW ROUTER — one view visible at a time, hash-deep-linked.
  // ==========================================================================
  var VIEWS = ["command", "arena", "agents", "memory", "system"];
  var currentView = "command";
  var loadedOnce = {};

  function showView(name, opts) {
    if (VIEWS.indexOf(name) === -1) {
      name = "command";
    }
    currentView = name;
    VIEWS.forEach(function (v) {
      var section = el("view-" + v);
      if (section) {
        var on = v === name;
        section.classList.toggle("is-active", on);
        section.hidden = !on;
      }
      var nav = el("nav-" + v);
      if (nav) {
        nav.setAttribute("aria-selected", v === name ? "true" : "false");
      }
    });
    // Keep the hash in sync for deep-linking + back-button (no reload).
    if (!opts || !opts.fromHash) {
      var want = "#" + name;
      if (window.location.hash !== want) {
        window.location.hash = want;
      }
    }
    // Lazy first-load for the heavier views.
    onViewEnter(name);
    // System is the only polling view; start/stop accordingly.
    syncSystemPolling();
  }

  function onViewEnter(name) {
    if (name === "agents" && !loadedOnce.agents) {
      loadedOnce.agents = true;
      loadRoster();
    }
    if (name === "arena" && !loadedOnce.arena) {
      loadedOnce.arena = true;
      loadModels();
    }
    if (name === "memory" && !loadedOnce.memory) {
      loadedOnce.memory = true;
      loadMemoryExtras();
    }
    if (name === "system" && !loadedOnce.system) {
      loadedOnce.system = true;
      refreshTelemetry();
      refreshTraces();
      refreshAudit();
    }
    if (name === "command") {
      var input = el("composer-input");
      if (input) {
        window.setTimeout(function () {
          input.focus();
        }, 0);
      }
    }
  }

  function wireRail() {
    VIEWS.forEach(function (v) {
      var nav = el("nav-" + v);
      if (nav) {
        nav.addEventListener("click", function () {
          showView(v);
        });
      }
    });
    window.addEventListener("hashchange", function () {
      var name = (window.location.hash || "").replace(/^#/, "") || "command";
      if (name !== currentView) {
        showView(name, { fromHash: true });
      }
    });
  }

  // --- chat helpers ----------------------------------------------------------
  /** Prefix a free-form question with the addressed persona, if any. */
  function withAddress(text) {
    if (addressTarget) {
      return addressTarget + ", " + text;
    }
    return text;
  }

  /** Send a turn to /chat; returns the parsed response (throws on HTTP error). */
  async function sendChat(text, opts) {
    var body = { session_id: SESSION_ID, text: text };
    if (opts && opts.confirmed) {
      body.confirmed = true;
    }
    if (activeModel) {
      body.model = activeModel;
    }
    return postJSON("/chat", body);
  }

  // A reply "needs confirmation" when the backend says so explicitly
  // (data.needs_confirmation), or when the in-character reply is the
  // orchestrator's confirm question (stable phrasing).
  function replyNeedsConfirmation(data) {
    if (!data) {
      return false;
    }
    if (data.needs_confirmation === true) {
      return true;
    }
    var text = String(data.text || "");
    return (
      /reply to confirm/i.test(text) ||
      /confirm before i act/i.test(text) ||
      /want me to go ahead\?/i.test(text)
    );
  }

  // ==========================================================================
  // COMMAND — centered conversation column + composer.
  // ==========================================================================
  function chatEmptyState() {
    var logNode = el("chat-log");
    if (!logNode || logNode.childNodes.length) {
      return;
    }
    var empty = mk("div", "chat-empty");
    empty.appendChild(mk("strong", null, "Good to see you."));
    empty.appendChild(
      document.createTextNode(
        "Ask anything, or address a specialist from the Agents view. Press ⌘K for the command palette."
      )
    );
    logNode.appendChild(empty);
  }

  function clearChatEmpty() {
    var logNode = el("chat-log");
    if (!logNode) {
      return;
    }
    var empty = logNode.querySelector(".chat-empty");
    if (empty) {
      logNode.removeChild(empty);
    }
  }

  function scrollChat() {
    var logNode = el("chat-log");
    if (logNode) {
      logNode.scrollTop = logNode.scrollHeight;
    }
  }

  function addUserTurn(text) {
    var logNode = el("chat-log");
    if (!logNode) {
      return null;
    }
    clearChatEmpty();
    var turn = mk("div", "turn user");
    turn.appendChild(mk("div", "turn-bubble", text));
    logNode.appendChild(turn);
    scrollChat();
    return turn;
  }

  /** Create a pending FRIDAY turn; returns it for later fill-in. */
  function addFridayPending() {
    var logNode = el("chat-log");
    if (!logNode) {
      return null;
    }
    var turn = mk("div", "turn friday pending");
    turn.appendChild(mk("div", "turn-bubble", "…"));
    logNode.appendChild(turn);
    scrollChat();
    return turn;
  }

  /** Fill a pending FRIDAY turn with its reply text + a route chip. */
  function fillFridayTurn(turn, data, isErr) {
    if (!turn) {
      return;
    }
    turn.classList.remove("pending");
    if (isErr) {
      turn.classList.add("err");
    }
    var bubble = turn.querySelector(".turn-bubble");
    if (bubble) {
      bubble.textContent = (data && data.text) || "(no reply)";
    }
    var route = data && data.route;
    if (route && !isErr) {
      var chip = mk("div", "route-chip");
      chip.appendChild(document.createTextNode((route.mode || data.mode || "—") + " · "));
      var agent = mk("b", null, route.agent || "FRIDAY");
      chip.appendChild(agent);
      if (route.confidence != null) {
        var pct = Math.round(Number(route.confidence) * 100);
        if (!isNaN(pct)) {
          chip.appendChild(document.createTextNode(" · "));
          chip.appendChild(mk("span", "rc-conf", pct + "%"));
        }
      }
      turn.appendChild(chip);
    }
    scrollChat();
  }

  /** A full chat round-trip from the composer/voice/palette. */
  async function submitChat(rawText) {
    var text = (rawText || "").trim();
    if (!text) {
      return;
    }
    var addressed = withAddress(text);
    addUserTurn(addressTarget ? "→ " + addressTarget + ": " + text : text);
    var pending = addFridayPending();
    try {
      var data = await sendChat(addressed);
      fillFridayTurn(pending, data, false);
      logLine("chat: " + text, "ok");
      maybeApproval(data, addressed);
    } catch (err) {
      fillFridayTurn(
        pending,
        { text: "Failed: " + (err && err.message ? err.message : err) },
        true
      );
      logLine("chat failed: " + text, "err");
    }
  }

  function autoGrow(area) {
    area.style.height = "auto";
    area.style.height = Math.min(area.scrollHeight, 200) + "px";
  }

  function wireComposer() {
    var form = el("composer");
    var input = el("composer-input");
    if (!form || !input) {
      return;
    }
    input.addEventListener("input", function () {
      autoGrow(input);
    });
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        var text = input.value;
        input.value = "";
        autoGrow(input);
        submitChat(text);
      }
    });
    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var text = input.value;
      input.value = "";
      autoGrow(input);
      submitChat(text);
    });
    chatEmptyState();
  }

  // --- persona addressing ----------------------------------------------------
  function setAddress(name) {
    addressTarget = name && name !== "FRIDAY" ? name : null;
    var chip = el("address-chip");
    var target = el("address-target");
    if (target) {
      target.textContent = addressTarget || "FRIDAY";
    }
    if (chip) {
      chip.hidden = !addressTarget;
    }
    // Reflect selection on the agent cards.
    var grid = el("agents-grid");
    if (grid) {
      var cards = grid.querySelectorAll(".agent-card");
      for (var i = 0; i < cards.length; i++) {
        cards[i].classList.toggle(
          "is-selected",
          cards[i].getAttribute("data-name") === addressTarget
        );
      }
    }
    if (addressTarget) {
      toast("Addressing " + addressTarget + " — type your message");
    }
  }

  function wireAddressClear() {
    var btn = el("address-clear");
    if (btn) {
      btn.addEventListener("click", function () {
        setAddress(null);
      });
    }
  }

  // ==========================================================================
  // AGENTS — roster cards (FRIDAY + specialists); click to address + jump.
  // ==========================================================================
  var rosterByName = {};
  var lastPersonas = [];

  var MODE_TO_PERSONA = {
    security: "EDITH",
    lockdown: "EDITH",
    device_control: "FORGE",
    automation: "ORACLE",
    scheduling: "ORACLE",
    schedule: "ORACLE",
    protocol: "ORACLE",
    reminder: "ORACLE",
    alerting: "KAREN",
    comms: "KAREN",
    communication: "KAREN",
    email: "KAREN",
    notify: "KAREN",
    finance: "GECKO",
    market: "GECKO",
    content: "VERONICA",
    outreach: "VERONICA",
    memory: "JOCASTA",
    knowledge: "JOCASTA",
    rag: "JOCASTA",
    graph: "JOCASTA",
    research: "VISION",
    analysis: "VISION",
    dev: "FORGE",
    development: "FORGE",
    system: "FORGE",
  };

  function personaForMode(mode) {
    if (!mode) {
      return "FRIDAY";
    }
    var key = String(mode).toLowerCase();
    if (MODE_TO_PERSONA[key]) {
      return MODE_TO_PERSONA[key];
    }
    for (var token in MODE_TO_PERSONA) {
      if (Object.prototype.hasOwnProperty.call(MODE_TO_PERSONA, token)) {
        if (key.indexOf(token) !== -1) {
          return MODE_TO_PERSONA[token];
        }
      }
    }
    return "FRIDAY";
  }

  function renderAgents() {
    var grid = el("agents-grid");
    if (!grid) {
      return;
    }
    var filterNode = el("agents-filter");
    var q = filterNode ? filterNode.value.trim().toLowerCase() : "";
    clear(grid);
    var shown = lastPersonas.filter(function (p) {
      if (!q) {
        return true;
      }
      return (
        String(p.name || "").toLowerCase().indexOf(q) !== -1 ||
        String(p.title || "").toLowerCase().indexOf(q) !== -1
      );
    });
    if (!shown.length) {
      grid.appendChild(mk("div", "panel-empty", q ? "no agents match" : "roster unavailable"));
      return;
    }
    shown.forEach(function (p) {
      var card = mk("button", "agent-card");
      card.type = "button";
      card.setAttribute("data-name", p.name);
      if (p.name === addressTarget) {
        card.classList.add("is-selected");
      }
      var head = mk("div", "agent-head");
      head.appendChild(mk("span", "agent-dot"));
      head.appendChild(mk("span", "agent-name", p.name));
      card.appendChild(head);
      card.appendChild(mk("div", "agent-title", p.title || ""));
      var scope = mk("div", "agent-scope");
      (p.scope || []).slice(0, 5).forEach(function (s) {
        scope.appendChild(mk("span", "scope-tag", s));
      });
      card.appendChild(scope);
      card.addEventListener("click", function () {
        setAddress(addressTarget === p.name ? null : p.name);
        if (addressTarget) {
          showView("command");
        }
      });
      grid.appendChild(card);
    });
    lightAgents(lastTraces);
  }

  async function loadRoster() {
    var data;
    try {
      data = await getJSON("/roster");
    } catch (err) {
      lastPersonas = [];
      renderAgents();
      return;
    }
    lastPersonas = (data && data.personas) || [];
    rosterByName = {};
    lastPersonas.forEach(function (p) {
      rosterByName[p.name] = p;
    });
    renderAgents();
    logLine("roster: " + lastPersonas.length + " personas", "ok");
  }

  /** Light agent cards whose mode a recent trace used; fade the rest. */
  function lightAgents(traces) {
    var grid = el("agents-grid");
    if (!grid) {
      return;
    }
    var active = {};
    (traces || []).slice(-6).forEach(function (t) {
      active[personaForMode(t.mode)] = true;
    });
    var cards = grid.querySelectorAll(".agent-card");
    for (var i = 0; i < cards.length; i++) {
      var name = cards[i].getAttribute("data-name");
      cards[i].classList.toggle("is-active", !!active[name]);
    }
  }

  function wireAgentsFilter() {
    var filterNode = el("agents-filter");
    if (filterNode) {
      filterNode.addEventListener("input", renderAgents);
    }
  }

  // ==========================================================================
  // BRAIN model picker — GET /models, POST /models/active. Graceful on 404.
  // ==========================================================================
  var lastModels = [];

  function setBrainLabel(text) {
    var label = el("brain-label");
    if (label) {
      label.textContent = text || "brain";
    }
  }

  async function loadModels() {
    var data;
    try {
      data = await getJSON("/models");
    } catch (err) {
      // 404 (no gateway) or unreachable: keep the pill quiet, arena shows empty.
      setBrainLabel("no gateway");
      lastModels = [];
      renderArenaModels();
      return;
    }
    lastModels = (data && data.models) || [];
    activeModel = (data && data.active) || activeModel;
    var act = lastModels.filter(function (m) {
      return m.id === activeModel;
    })[0];
    setBrainLabel(act ? act.label || act.model || act.id : activeModel || "brain");
    renderArenaModels();
  }

  function renderBrainList() {
    var listNode = el("brain-list");
    if (!listNode) {
      return;
    }
    clear(listNode);
    if (!lastModels.length) {
      listNode.appendChild(
        mk("div", "panel-empty", "No model gateway available on this backend.")
      );
      return;
    }
    lastModels.forEach(function (m) {
      var row = mk("button", "brain-row");
      row.type = "button";
      if (m.id === activeModel) {
        row.classList.add("is-active");
      }
      var left = mk("div");
      left.appendChild(mk("div", "br-label", m.label || m.model || m.id));
      left.appendChild(mk("div", "br-prov", (m.provider || "") + " · " + (m.model || m.id)));
      row.appendChild(left);
      var right = mk("div");
      if (m.free) {
        right.appendChild(mk("span", "br-free", "free"));
      }
      if (m.id === activeModel) {
        right.appendChild(mk("span", "br-active-tag", " active"));
      }
      row.appendChild(right);
      row.addEventListener("click", function () {
        chooseModel(m);
      });
      listNode.appendChild(row);
    });
  }

  async function chooseModel(m) {
    try {
      var resp = await postJSON("/models/active", { model_id: m.id });
      activeModel = (resp && resp.active) || m.id;
      setBrainLabel(m.label || m.model || m.id);
      renderBrainList();
      toast("Brain set to " + (m.label || m.id));
      logLine("brain: " + (m.label || m.id), "ok");
      closeBrain();
    } catch (err) {
      toast("Could not switch model: " + (err && err.message ? err.message : err));
    }
  }

  function openBrain() {
    var overlay = el("brain-overlay");
    if (!overlay) {
      return;
    }
    renderBrainList();
    overlay.hidden = false;
  }
  function closeBrain() {
    var overlay = el("brain-overlay");
    if (overlay) {
      overlay.hidden = true;
    }
  }
  function wireBrain() {
    var pill = el("brain-pill");
    if (pill) {
      pill.addEventListener("click", openBrain);
    }
    var close = el("brain-close");
    if (close) {
      close.addEventListener("click", closeBrain);
    }
    var overlay = el("brain-overlay");
    if (overlay) {
      overlay.addEventListener("click", function (ev) {
        if (ev.target === overlay) {
          closeBrain();
        }
      });
    }
  }

  // ==========================================================================
  // ARENA — POST /models/compare. Multi-select, judge toggle, result grid.
  // ==========================================================================
  function renderArenaModels() {
    var host = el("arena-models");
    if (!host) {
      return;
    }
    clear(host);
    if (!lastModels.length) {
      host.appendChild(
        mk("div", "panel-empty", "No models to compare — model gateway is off.")
      );
      return;
    }
    lastModels.forEach(function (m, idx) {
      var chip = mk("label", "model-chip");
      var box = mk("input");
      box.type = "checkbox";
      box.value = m.id;
      box.checked = idx < 4; // first ~4 checked by default
      if (box.checked) {
        chip.classList.add("is-on");
      }
      box.addEventListener("change", function () {
        chip.classList.toggle("is-on", box.checked);
      });
      chip.appendChild(box);
      chip.appendChild(mk("span", null, m.label || m.model || m.id));
      if (m.free) {
        chip.appendChild(mk("span", "mc-free", "free"));
      }
      host.appendChild(chip);
    });
  }

  function selectedArenaModels() {
    var host = el("arena-models");
    if (!host) {
      return [];
    }
    var boxes = host.querySelectorAll('input[type="checkbox"]');
    var ids = [];
    for (var i = 0; i < boxes.length; i++) {
      if (boxes[i].checked) {
        ids.push(boxes[i].value);
      }
    }
    return ids;
  }

  function labelForModel(id) {
    var m = lastModels.filter(function (x) {
      return x.id === id;
    })[0];
    return m ? m.label || m.model || m.id : id;
  }
  function isFreeModel(id) {
    var m = lastModels.filter(function (x) {
      return x.id === id;
    })[0];
    return !!(m && m.free);
  }

  function renderArenaLoading(ids) {
    var results = el("arena-results");
    if (!results) {
      return;
    }
    clear(results);
    ids.forEach(function (id) {
      var card = mk("div", "arena-card loading");
      card.setAttribute("data-model", id);
      var head = mk("div", "arena-card-head");
      head.appendChild(mk("span", "ac-label", labelForModel(id)));
      if (isFreeModel(id)) {
        head.appendChild(mk("span", "ac-free", "free"));
      }
      card.appendChild(head);
      card.appendChild(mk("div", "ac-text", "thinking…"));
      results.appendChild(card);
    });
  }

  function renderArenaResults(payload) {
    var results = el("arena-results");
    if (!results) {
      return;
    }
    clear(results);
    var rows = (payload && payload.results) || [];
    var best = payload && payload.best;
    if (!rows.length) {
      results.appendChild(mk("div", "panel-empty", "no results"));
      return;
    }
    rows.forEach(function (r) {
      var card = mk("div", "arena-card");
      if (r.model_id === best) {
        card.classList.add("winner");
      }
      if (r.ok === false) {
        card.classList.add("err");
      }
      var head = mk("div", "arena-card-head");
      head.appendChild(mk("span", "ac-label", r.label || labelForModel(r.model_id)));
      if (isFreeModel(r.model_id)) {
        head.appendChild(mk("span", "ac-free", "free"));
      }
      if (r.model_id === best) {
        head.appendChild(mk("span", "ac-best", "🏆 BEST"));
      }
      card.appendChild(head);
      var text =
        r.ok === false
          ? "error: " + (r.error || "failed")
          : r.text || "(empty)";
      card.appendChild(mk("div", "ac-text", text));
      var foot = mk("div", "ac-foot");
      foot.appendChild(mk("span", null, r.ok === false ? "failed" : "ok"));
      var lat = r.latency_ms != null ? (r.latency_ms / 1000).toFixed(1) + "s" : "—";
      foot.appendChild(mk("span", "ac-lat", lat));
      card.appendChild(foot);
      results.appendChild(card);
    });
  }

  async function runArena() {
    var promptNode = el("arena-prompt");
    var judgeNode = el("arena-judge");
    var go = el("arena-go");
    var prompt = promptNode ? promptNode.value.trim() : "";
    if (!prompt) {
      toast("Type a prompt for the arena first");
      return;
    }
    var ids = selectedArenaModels();
    if (!ids.length) {
      toast("Pick at least one model");
      return;
    }
    if (go) {
      go.disabled = true;
    }
    renderArenaLoading(ids);
    try {
      var payload = await postJSON("/models/compare", {
        prompt: prompt,
        models: ids,
        judge: judgeNode ? !!judgeNode.checked : true,
      });
      renderArenaResults(payload);
      logLine("arena: " + ids.length + " models", "ok");
    } catch (err) {
      var results = el("arena-results");
      if (results) {
        clear(results);
        results.appendChild(
          mk("div", "panel-empty", "Arena failed: " + (err && err.message ? err.message : err))
        );
      }
      logLine("arena failed", "err");
    } finally {
      if (go) {
        go.disabled = false;
      }
    }
  }

  function wireArena() {
    var go = el("arena-go");
    if (go) {
      go.addEventListener("click", runArena);
    }
  }

  // ==========================================================================
  // MEMORY — dossier search + optional RAG/graph/briefing/journal subsections.
  // ==========================================================================
  async function runDossier(query) {
    var q = (query || "").trim();
    var cards = el("dossier-cards");
    if (!q || !cards) {
      return;
    }
    var card = mk("div", "card dossier loading");
    var head = mk("div", "card-head");
    head.appendChild(mk("span", "card-title", q));
    head.appendChild(mk("span", "card-meta", "querying…"));
    card.appendChild(head);
    var bodyNode = mk("div", "card-body", "FRIDAY is compiling the dossier…");
    card.appendChild(bodyNode);
    cards.insertBefore(card, cards.firstChild);
    while (cards.childNodes.length > 6) {
      cards.removeChild(cards.lastChild);
    }
    try {
      var data = await sendChat(withAddress("Give me a dossier on " + q));
      card.classList.remove("loading");
      var meta = card.querySelector(".card-meta");
      if (meta) {
        meta.textContent = data && data.mode ? data.mode : "ready";
      }
      bodyNode.textContent = (data && data.text) || "(no dossier returned)";
      maybeApproval(data, "Give me a dossier on " + q);
      logLine("dossier: " + q, "ok");
    } catch (err) {
      card.classList.remove("loading");
      card.classList.add("err");
      var meta2 = card.querySelector(".card-meta");
      if (meta2) {
        meta2.textContent = "error";
      }
      bodyNode.textContent =
        "Dossier failed: " + (err && err.message ? err.message : err);
    }
  }

  function wireDossier() {
    var form = el("dossier-form");
    var input = el("dossier-input");
    if (!form || !input) {
      return;
    }
    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      runDossier(input.value);
      input.value = "";
    });
  }

  /** Load the optional Memory subsections; each hides gracefully on error. */
  function loadMemoryExtras() {
    loadRagSources();
    loadGraphEntities();
    loadBriefing();
    loadJournal();
  }

  async function loadRagSources() {
    var panel = el("rag-panel");
    var host = el("rag-sources");
    if (!panel || !host) {
      return;
    }
    var data;
    try {
      data = await getJSON("/rag/sources");
    } catch (err) {
      panel.hidden = true; // 404 => RAG off
      return;
    }
    panel.hidden = false;
    var sources = (data && data.sources) || [];
    clear(host);
    if (!sources.length) {
      host.appendChild(mk("div", "panel-empty", "No sources yet — drop a file to ingest."));
      return;
    }
    sources.forEach(function (id) {
      var row = mk("div", "rag-row");
      row.appendChild(mk("span", "rag-id", id));
      var del = mk("button", "rag-del", "delete");
      del.type = "button";
      del.addEventListener("click", function () {
        deleteRagSource(id, row);
      });
      row.appendChild(del);
      host.appendChild(row);
    });
  }

  async function deleteRagSource(id, row) {
    try {
      var resp = await fetch(url("/rag/sources/" + encodeURIComponent(id)), {
        method: "DELETE",
      });
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status);
      }
      if (row && row.parentNode) {
        row.parentNode.removeChild(row);
      }
      toast("Removed " + id);
      logLine("rag source removed: " + id, "ok");
    } catch (err) {
      toast("Delete failed: " + (err && err.message ? err.message : err));
    }
  }

  async function loadGraphEntities() {
    var panel = el("graph-panel");
    var host = el("graph-entities");
    if (!panel || !host) {
      return;
    }
    var data;
    try {
      data = await getJSON("/graph/entities");
    } catch (err) {
      panel.hidden = true;
      return;
    }
    var entities = (data && data.entities) || [];
    if (!entities.length) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    clear(host);
    entities.slice(0, 60).forEach(function (e) {
      var name = typeof e === "string" ? e : e.name || e.id || JSON.stringify(e);
      host.appendChild(mk("span", "entity-chip", name));
    });
  }

  async function loadBriefing() {
    var panel = el("briefing-panel");
    var body = el("briefing-body");
    if (!panel || !body) {
      return;
    }
    var data;
    try {
      data = await getJSON("/briefing");
    } catch (err) {
      panel.hidden = true;
      return;
    }
    var text =
      data && typeof data === "object"
        ? data.text || data.summary || JSON.stringify(data, null, 2)
        : String(data);
    if (!text || !text.trim()) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    body.textContent = text;
  }

  async function loadJournal() {
    var panel = el("journal-panel");
    var host = el("journal-list");
    if (!panel || !host) {
      return;
    }
    var data;
    try {
      data = await getJSON("/journal");
    } catch (err) {
      panel.hidden = true;
      return;
    }
    var entries = (data && data.entries) || [];
    if (!entries.length) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    clear(host);
    entries.slice(0, 12).forEach(function (e) {
      var text =
        typeof e === "string" ? e : e.text || e.summary || JSON.stringify(e);
      host.appendChild(mk("div", "journal-row", text));
    });
  }

  // ==========================================================================
  // APPROVALS — inline in the chat log + a toast. Approve re-sends confirmed.
  // ==========================================================================
  function maybeApproval(data, originalText) {
    if (!replyNeedsConfirmation(data)) {
      return false;
    }
    var logNode = el("chat-log");
    if (!logNode) {
      return false;
    }
    clearChatEmpty();

    var card = mk("div", "approval");
    card.appendChild(mk("div", "ap-title", "Confirmation required"));
    card.appendChild(mk("div", "ap-body", (data && data.text) || ""));

    var detail = data && (data.confirmation || data.pending || data.tool);
    if (detail && typeof detail === "object") {
      var pre = mk("div", "ap-detail");
      if (detail.tool) {
        pre.appendChild(mk("div", "kv", "tool: " + detail.tool));
      }
      var args = detail.args || detail.arguments;
      if (args && typeof args === "object") {
        Object.keys(args).forEach(function (k) {
          pre.appendChild(mk("div", "kv", k + ": " + String(args[k])));
        });
      }
      card.appendChild(pre);
    }

    var actions = mk("div", "ap-actions");
    var approve = mk("button", "btn approve", "Approve");
    approve.type = "button";
    var deny = mk("button", "btn deny", "Deny");
    deny.type = "button";

    approve.addEventListener("click", async function () {
      approve.disabled = true;
      deny.disabled = true;
      try {
        var confirmed = await sendChat(originalText, { confirmed: true });
        card.classList.add("resolved");
        var body = card.querySelector(".ap-body");
        if (body) {
          body.textContent = (confirmed && confirmed.text) || "Done.";
        }
        logLine("approved: " + originalText, "ok");
        // Chained confirmation: re-card if the confirmed turn still needs it.
        maybeApproval(confirmed, originalText);
      } catch (err) {
        approve.disabled = false;
        deny.disabled = false;
        toast("Approve failed: " + (err && err.message ? err.message : err));
      }
    });

    deny.addEventListener("click", function () {
      card.classList.add("denied");
      approve.disabled = true;
      deny.disabled = true;
      logLine("denied: " + originalText);
      // Recoverable: offer an undo that re-opens the same approval.
      var undo = mk("button", "btn undo", "Undo");
      undo.type = "button";
      undo.addEventListener("click", function () {
        if (card.parentNode) {
          card.parentNode.removeChild(card);
        }
        maybeApproval(data, originalText);
      });
      actions.appendChild(undo);
    });

    actions.appendChild(approve);
    actions.appendChild(deny);
    card.appendChild(actions);

    logNode.appendChild(card);
    scrollChat();
    toast("FRIDAY needs your confirmation");
    return true;
  }

  // ==========================================================================
  // SYSTEM — telemetry / by-mode / audit / traces / replay / log / host stats.
  // ==========================================================================
  var lastTraces = [];
  var lastAudit = { tool_calls: [], security: [] };

  function spanDurationMs(span) {
    if (span && typeof span.start === "number" && typeof span.end === "number") {
      var ms = (span.end - span.start) * 1000;
      return ms >= 0 ? ms : 0;
    }
    if (span && typeof span.ms === "number") {
      return span.ms;
    }
    return null;
  }

  function renderByMode(byMode) {
    var host = el("by-mode");
    if (!host) {
      return;
    }
    clear(host);
    var entries = Object.keys(byMode || {});
    if (!entries.length) {
      host.appendChild(mk("div", "panel-empty", "—"));
      return;
    }
    var max = 1;
    entries.forEach(function (k) {
      if (byMode[k] > max) {
        max = byMode[k];
      }
    });
    entries
      .sort(function (a, b) {
        return byMode[b] - byMode[a];
      })
      .forEach(function (k) {
        var row = mk("div", "mode-row");
        row.appendChild(mk("span", "mode-name", k));
        var barWrap = mk("span", "mode-bar");
        var fill = mk("i");
        fill.style.width = Math.round((byMode[k] / max) * 100) + "%";
        barWrap.appendChild(fill);
        row.appendChild(barWrap);
        row.appendChild(mk("span", "mode-count", String(byMode[k])));
        host.appendChild(row);
      });
  }

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
      renderByMode(metrics.by_mode || {});
      setOnline(true);
    } catch (err) {
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
      /* state best-effort */
    }
    loadSystemStats();
  }

  async function loadSystemStats() {
    var panel = el("sysstats-panel");
    var host = el("sysstats");
    var alerts = el("sysalerts");
    if (!panel || !host) {
      return;
    }
    var stats;
    try {
      stats = await getJSON("/system/stats");
    } catch (err) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    clear(host);
    ["cpu", "mem", "disk"].forEach(function (key) {
      if (stats[key] == null) {
        return;
      }
      var raw = Number(stats[key]);
      var pct = isNaN(raw) ? 0 : raw <= 1 ? raw * 100 : raw;
      pct = Math.max(0, Math.min(100, Math.round(pct)));
      var row = mk("div", "sys-row");
      row.appendChild(mk("span", "sys-name", key));
      var bar = mk("span", "sys-bar");
      var fill = mk("i");
      fill.style.width = pct + "%";
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(mk("span", "sys-val", pct + "%"));
      host.appendChild(row);
    });
    if (alerts) {
      clear(alerts);
    }
    try {
      var check = await getJSON("/system/check");
      var list = (check && check.alerts) || [];
      if (alerts) {
        list.slice(0, 6).forEach(function (a) {
          var text = typeof a === "string" ? a : a.message || JSON.stringify(a);
          alerts.appendChild(mk("div", "sys-alert", text));
        });
      }
    } catch (err) {
      /* alerts optional */
    }
  }

  function renderAudit() {
    var list = el("audit-list");
    if (!list) {
      return;
    }
    clear(list);
    var rows = (lastAudit.tool_calls || []).slice(-10).reverse();
    if (!rows.length) {
      list.appendChild(mk("div", "panel-empty", "no tool calls yet"));
      return;
    }
    rows.forEach(function (r) {
      var item = mk("div", "audit-row " + (r.ok ? "ok" : "err"));
      item.appendChild(mk("span", "audit-tool", r.tool));
      item.appendChild(
        mk("span", "audit-ok", r.ok ? "✓" : "✕ " + (r.error_code || ""))
      );
      list.appendChild(item);
    });
  }

  async function refreshAudit() {
    try {
      var data = await getJSON("/admin/audit");
      lastAudit = {
        tool_calls: (data && data.tool_calls) || [],
        security: (data && data.security) || [],
      };
      renderAudit();
    } catch (err) {
      /* audit best-effort */
    }
    var badge = el("audit-verify");
    if (!badge) {
      return;
    }
    try {
      var v = await getJSON("/admin/audit/verify");
      var ok = v && v.ok !== false;
      badge.textContent = ok ? "verified" : "TAMPERED";
      badge.classList.toggle("is-ok", ok);
      badge.classList.toggle("is-bad", !ok);
    } catch (err) {
      badge.textContent = "n/a";
      badge.classList.remove("is-ok", "is-bad");
    }
  }

  function renderFlows() {
    var list = el("flow-list");
    if (!list) {
      return;
    }
    clear(list);
    if (!lastTraces.length) {
      list.appendChild(mk("div", "panel-empty", "no traces yet"));
      return;
    }
    var traces = lastTraces.slice().reverse(); // newest first
    traces.forEach(function (t) {
      var row = mk("button", "flow");
      row.type = "button";
      row.title = "Replay " + t.correlation_id;
      var head = mk("span", "flow-head");
      head.appendChild(mk("span", "flow-mode", t.mode || "—"));
      head.appendChild(mk("span", "flow-cid", String(t.correlation_id).slice(0, 8)));
      row.appendChild(head);

      var pipe = mk("span", "flow-pipe");
      var spans = t.spans || [];
      if (!spans.length) {
        pipe.appendChild(mk("span", "flow-step empty", "—"));
      }
      spans.forEach(function (s, idx) {
        if (idx > 0) {
          pipe.appendChild(mk("span", "flow-arrow", "→"));
        }
        var step = mk("span", "flow-step", s.name || "span");
        var ms = spanDurationMs(s);
        if (ms != null) {
          step.appendChild(mk("span", "flow-ms", " " + Math.round(ms) + "ms"));
        }
        pipe.appendChild(step);
      });
      row.appendChild(pipe);
      row.addEventListener("click", function () {
        openReplay(t);
      });
      list.appendChild(row);
    });
  }

  function openReplay(trace) {
    var overlay = el("replay-overlay");
    var idNode = el("replay-id");
    var spansNode = el("replay-spans");
    var auditNode = el("replay-audit");
    if (!overlay || !spansNode || !auditNode) {
      return;
    }
    overlay.hidden = false;
    if (idNode) {
      idNode.textContent = String(trace.correlation_id).slice(0, 12);
    }
    clear(spansNode);
    (trace.spans || []).forEach(function (s) {
      var item = mk("div", "rspan");
      item.appendChild(mk("span", "rspan-name", s.name || "span"));
      var ms = spanDurationMs(s);
      item.appendChild(
        mk("span", "rspan-ms", ms != null ? Math.round(ms) + "ms" : (s.ok === false ? "fail" : "open"))
      );
      var attrs = s.attrs || {};
      var keys = Object.keys(attrs);
      if (keys.length) {
        var meta = keys
          .slice(0, 4)
          .map(function (k) {
            return k + "=" + String(attrs[k]);
          })
          .join("  ");
        item.appendChild(mk("span", "rspan-attrs", meta));
      }
      spansNode.appendChild(item);
    });
    clear(auditNode);
    var rows = (lastAudit.tool_calls || []).filter(function (r) {
      return r.correlation_id === trace.correlation_id;
    });
    if (!rows.length) {
      auditNode.appendChild(mk("div", "panel-empty", "no matching audit rows"));
    } else {
      rows.forEach(function (r) {
        var item = mk("div", "arow " + (r.ok ? "ok" : "err"));
        item.appendChild(mk("span", "arow-tool", r.tool));
        var args = r.args_redacted || {};
        var argStr = Object.keys(args)
          .map(function (k) {
            return k + "=" + String(args[k]);
          })
          .join(" ");
        item.appendChild(mk("span", "arow-args", argStr || "(no args)"));
        item.appendChild(
          mk("span", "arow-ok", r.ok ? "ok" : "fail:" + (r.error_code || "?"))
        );
        auditNode.appendChild(item);
      });
    }
    logLine("replay " + String(trace.correlation_id).slice(0, 8));
  }

  function closeReplay() {
    var overlay = el("replay-overlay");
    if (overlay) {
      overlay.hidden = true;
    }
  }

  function wireReplay() {
    var btn = el("replay-close");
    if (btn) {
      btn.addEventListener("click", closeReplay);
    }
    var overlay = el("replay-overlay");
    if (overlay) {
      overlay.addEventListener("click", function (ev) {
        if (ev.target === overlay) {
          closeReplay();
        }
      });
    }
  }

  async function refreshTraces() {
    try {
      var data = await getJSON("/admin/traces");
      lastTraces = (data && data.traces) || [];
      var tag = el("flow-tag");
      if (tag) {
        tag.textContent = lastTraces.length ? "live" : "idle";
      }
      renderFlows();
      lightAgents(lastTraces);
    } catch (err) {
      var tag2 = el("flow-tag");
      if (tag2) {
        tag2.textContent = "offline";
      }
    }
  }

  // --- System polling: ONLY while System is the active view -----------------
  var systemTimers = [];
  function syncSystemPolling() {
    if (currentView === "system") {
      if (systemTimers.length) {
        return;
      }
      systemTimers.push(window.setInterval(refreshTelemetry, 5000));
      systemTimers.push(window.setInterval(refreshTraces, 4000));
      systemTimers.push(window.setInterval(refreshAudit, 8000));
    } else {
      systemTimers.forEach(function (t) {
        window.clearInterval(t);
      });
      systemTimers = [];
    }
  }

  // ==========================================================================
  // GLOBE / VOICE / DRAG-DROP.
  // ==========================================================================
  function wireGlobe() {
    var btn = el("btn-globe");
    if (btn) {
      btn.addEventListener("click", function () {
        window.open(url("/maps"), "_blank", "noopener");
        logLine("opened /maps");
      });
    }
  }

  // ==========================================================================
  // VOICE — wake/summon: reveal the cockpit + speak each operator's own voice.
  // ==========================================================================
  var VOICE_STYLES = {
    FRIDAY: { pitch: 1.15, rate: 1.02, hint: "female" },
    EDITH: { pitch: 0.9, rate: 1.08, hint: "female" },
    ORACLE: { pitch: 1.0, rate: 0.92, hint: "female" },
    GECKO: { pitch: 0.7, rate: 1.0, hint: "male" },
    KAREN: { pitch: 1.3, rate: 1.12, hint: "female" },
    VERONICA: { pitch: 1.08, rate: 1.05, hint: "female" },
    JOCASTA: { pitch: 0.95, rate: 0.88, hint: "female" },
    VISION: { pitch: 0.85, rate: 0.98, hint: "male" },
    FORGE: { pitch: 0.62, rate: 1.0, hint: "male" },
  };
  var WAKE_OPS = Object.keys(VOICE_STYLES);

  function pickVoice(hint) {
    var synth = window.speechSynthesis;
    if (!synth) return null;
    var voices = synth.getVoices() || [];
    var h = (hint || "").toLowerCase();
    for (var i = 0; i < voices.length; i++) {
      if (h && voices[i].name.toLowerCase().indexOf(h) !== -1) return voices[i];
    }
    return null;
  }

  function speak(text, operator) {
    var synth = window.speechSynthesis;
    if (!synth || !text) return;
    var style =
      VOICE_STYLES[(operator || "FRIDAY").toUpperCase()] ||
      { pitch: 1, rate: 1, hint: "" };
    var u = new SpeechSynthesisUtterance(text);
    u.pitch = style.pitch;
    u.rate = style.rate;
    var v = pickVoice(style.hint);
    if (v) u.voice = v;
    synth.cancel();
    synth.speak(u);
  }

  // Client-side mirror of friday.voice.wake.parse_wake_command (push-to-talk path).
  function parseWake(text) {
    var t = (text || "").trim().toLowerCase();
    if (!t) return null;
    var sum = t.match(
      /\bfrid?ay\s*,?\s+(?:summon|spawn|call|get|bring up|wake)\s+([a-z]+)\b/
    );
    if (sum) {
      var name = sum[1].toUpperCase();
      if (WAKE_OPS.indexOf(name) !== -1) {
        return { type: "summon", operator: name, greeting: name + " here, Boss." };
      }
      return null;
    }
    if (/\b(?:hey|hi|ok|okay)\s*,?\s+frid?ay\b/.test(t)) {
      return { type: "wake", operator: "FRIDAY", greeting: "I'm up, Boss." };
    }
    return null;
  }

  // React to a wake/summon: reveal the cockpit, address the operator, speak.
  function handleWake(evt) {
    if (!evt || !evt.type || evt.type === "none") return;
    showView("command");
    if (evt.type === "summon" && evt.operator) {
      setAddress(evt.operator);
      toast("Summoned " + evt.operator);
    }
    speak(evt.greeting || "I'm up, Boss.", evt.operator || "FRIDAY");
    logLine("wake: " + (evt.operator || "FRIDAY") + " — " + (evt.greeting || ""), "ok");
  }

  function wakeWsUrl() {
    var base = resolveApiBase();
    if (base && /^https?:/.test(base)) return base.replace(/^http/, "ws") + "/ws/wake";
    var scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    return scheme + "//" + window.location.host + "/ws/wake";
  }

  // Server-side always-on path: react to wake/summon events pushed over /ws/wake
  // (a server STT runner feeds the wake engine). Inert when the wake word is off
  // (the socket is refused) — the browser push-to-talk path still works.
  function connectWake() {
    if (!("WebSocket" in window)) return;
    var ws;
    try {
      ws = new WebSocket(wakeWsUrl());
    } catch (err) {
      return;
    }
    ws.addEventListener("message", function (ev) {
      try {
        handleWake(JSON.parse(ev.data));
      } catch (err) {
        /* ignore ready / non-JSON frames */
      }
    });
  }

  function wireVoice() {
    var btn = el("btn-mic");
    if (!btn) {
      return;
    }
    var Rec = window.SpeechRecognition || window.webkitSpeechRecognition || null;
    if (!Rec) {
      btn.addEventListener("click", function () {
        toast("Voice not supported in this browser");
      });
      btn.classList.add("is-disabled");
      return;
    }
    var rec = new Rec();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    var listening = false;

    rec.addEventListener("result", function (ev) {
      var said = "";
      try {
        said = ev.results[0][0].transcript;
      } catch (err) {
        said = "";
      }
      if (!said) {
        return;
      }
      logLine("voice: " + said);
      var wake = parseWake(said);
      if (wake) {
        handleWake(wake);
      } else {
        showView("command");
        submitChat(said);
      }
    });
    rec.addEventListener("end", function () {
      listening = false;
      btn.classList.remove("is-listening");
    });
    rec.addEventListener("error", function (ev) {
      listening = false;
      btn.classList.remove("is-listening");
      toast("Voice error: " + (ev && ev.error ? ev.error : "unknown"));
    });

    btn.addEventListener("click", function () {
      if (listening) {
        rec.stop();
        return;
      }
      try {
        rec.start();
        listening = true;
        btn.classList.add("is-listening");
        toast("Listening…");
      } catch (err) {
        toast("Could not start voice: " + (err && err.message ? err.message : err));
      }
    });
  }

  async function ingestFile(file) {
    var fd = new FormData();
    fd.append("file", file, file.name);
    toast("Ingesting " + file.name + "…");
    try {
      var resp = await fetch(url("/rag/ingest"), { method: "POST", body: fd });
      if (resp.status === 404) {
        toast("RAG is disabled on this backend");
        logLine("rag ingest skipped (disabled): " + file.name, "err");
        return;
      }
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status);
      }
      var data = await resp.json();
      var src = (data && data.source_id) || file.name;
      var chunks = data && data.chunks != null ? data.chunks : "?";
      toast("Ingested " + src + " (" + chunks + " chunks) — now askable");
      logLine("ingested " + src + " — " + chunks + " chunks", "ok");
      loadRagSources();
    } catch (err) {
      toast("Ingest failed: " + (err && err.message ? err.message : err));
      logLine("ingest failed: " + file.name, "err");
    }
  }

  function wireDragDrop() {
    var overlay = el("drop-overlay");
    var depth = 0;

    function hasFiles(ev) {
      return (
        ev.dataTransfer &&
        Array.prototype.indexOf.call(ev.dataTransfer.types || [], "Files") !== -1
      );
    }

    window.addEventListener("dragenter", function (ev) {
      if (hasFiles(ev)) {
        ev.preventDefault();
        depth++;
        if (overlay) {
          overlay.classList.add("is-active");
        }
      }
    });
    window.addEventListener("dragover", function (ev) {
      if (hasFiles(ev)) {
        ev.preventDefault();
        ev.dataTransfer.dropEffect = "copy";
      }
    });
    window.addEventListener("dragleave", function () {
      depth = Math.max(0, depth - 1);
      if (depth === 0 && overlay) {
        overlay.classList.remove("is-active");
      }
    });
    window.addEventListener("drop", function (ev) {
      depth = 0;
      if (overlay) {
        overlay.classList.remove("is-active");
      }
      if (!ev.dataTransfer || !ev.dataTransfer.files || !ev.dataTransfer.files.length) {
        return;
      }
      ev.preventDefault();
      var files = ev.dataTransfer.files;
      for (var i = 0; i < files.length; i++) {
        ingestFile(files[i]);
      }
    });
  }

  // ==========================================================================
  // THEMES — re-skin via data-theme on <html>; choice persists in localStorage.
  // ==========================================================================
  var THEMES = ["default", "amber", "crimson", "emerald", "light"];
  var THEME_KEY = "friday-theme";

  function applyTheme(name) {
    var theme = THEMES.indexOf(name) >= 0 ? name : "default";
    if (theme === "default") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
    try {
      window.localStorage.setItem(THEME_KEY, theme);
    } catch (e) {
      /* localStorage may be unavailable (private mode) — theme still applies */
    }
    return theme;
  }

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || "default";
  }

  function cycleTheme() {
    var idx = THEMES.indexOf(currentTheme());
    return applyTheme(THEMES[(idx + 1) % THEMES.length]);
  }

  function applyStoredTheme() {
    var saved = "default";
    try {
      saved = window.localStorage.getItem(THEME_KEY) || "default";
    } catch (e) {
      /* ignore */
    }
    applyTheme(saved);
  }

  // ==========================================================================
  // COMMAND PALETTE (Cmd/Ctrl-K) — actions + view switches.
  // ==========================================================================
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
        showView("command");
        submitChat(q);
        closePalette();
        return "Sent to FRIDAY.";
      },
    },
    {
      id: "dossier",
      title: "Open dossier",
      hint: "POST /chat",
      glyph: "❒",
      run: async function (query) {
        var q = (query || "").trim();
        if (!q) {
          return "Type a person/project after the command.";
        }
        showView("memory");
        runDossier(q);
        closePalette();
        return "Dossier requested for: " + q;
      },
    },
    { id: "view-command", title: "Go to Command", hint: "view", glyph: "◉", run: async function () { showView("command"); closePalette(); return ""; } },
    { id: "view-arena", title: "Go to Arena", hint: "view", glyph: "⚔", run: async function () { showView("arena"); closePalette(); return ""; } },
    { id: "view-agents", title: "Go to Agents", hint: "view", glyph: "❖", run: async function () { showView("agents"); closePalette(); return ""; } },
    { id: "view-memory", title: "Go to Memory", hint: "view", glyph: "▤", run: async function () { showView("memory"); closePalette(); return ""; } },
    { id: "view-system", title: "Go to System", hint: "view", glyph: "⌁", run: async function () { showView("system"); closePalette(); return ""; } },
    { id: "theme", title: "Cycle theme", hint: "appearance", glyph: "◐", run: async function () { var t = cycleTheme(); closePalette(); return "Theme: " + t; } },
    {
      id: "roster",
      title: "Show roster",
      hint: "GET /roster",
      glyph: "❖",
      run: async function () {
        var r = await getJSON("/roster");
        return (r.personas || [])
          .map(function (p) {
            return p.name + " — " + p.title;
          })
          .join("\n");
      },
    },
    {
      id: "traces",
      title: "Show traces",
      hint: "GET /admin/traces",
      glyph: "⌁",
      run: async function () {
        var t = await getJSON("/admin/traces");
        var traces = t.traces || [];
        return (
          traces.length +
          " trace(s)\n" +
          traces
            .slice(-8)
            .map(function (x) {
              return (
                String(x.correlation_id).slice(0, 8) +
                " [" +
                (x.mode || "—") +
                "] " +
                (x.spans || [])
                  .map(function (s) {
                    return s.name;
                  })
                  .join(" → ")
              );
            })
            .join("\n")
        );
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
      id: "audit",
      title: "Show audit",
      hint: "GET /admin/audit",
      glyph: "▤",
      run: async function () {
        var a = await getJSON("/admin/audit");
        var v = await getJSON("/admin/audit/verify").catch(function () {
          return { ok: null };
        });
        var rows = a.tool_calls || [];
        return (
          "verify: " +
          (v.ok === false ? "TAMPERED" : v.ok === true ? "ok" : "n/a") +
          "\n" +
          rows
            .slice(-12)
            .map(function (r) {
              return (
                (r.ok ? "✓ " : "✕ ") +
                r.tool +
                (r.error_code ? " (" + r.error_code + ")" : "")
              );
            })
            .join("\n")
        );
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
    {
      id: "globe",
      title: "Open globe",
      hint: "GET /maps",
      glyph: "◍",
      run: async function () {
        window.open(url("/maps"), "_blank", "noopener");
        return "Opening the 3D globe…";
      },
    },
  ];

  var paletteState = { open: false, active: 0, filtered: COMMANDS.slice() };

  function isFreeQuery(value) {
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
    clear(list);
    paletteState.filtered.forEach(function (cmd, idx) {
      var li = mk("li", idx === paletteState.active ? "is-active" : "");
      li.setAttribute("role", "option");
      li.appendChild(mk("span", "cmd-glyph", cmd.glyph || "▸"));
      li.appendChild(mk("span", "cmd-title", cmd.title));
      li.appendChild(mk("span", "cmd-hint", cmd.hint || ""));
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
      var takesArg = cmd.id === "ask" || cmd.id === "dossier";
      var arg = takesArg ? raw : "";
      var out = await cmd.run(arg);
      showResult(out, false);
    } catch (err) {
      logLine(String(err && err.message ? err.message : err), "err");
      showResult(
        "Command failed: " + (err && err.message ? err.message : err),
        true
      );
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
        return;
      }
      if (ev.key === "Escape") {
        // Close any open overlay from anywhere (not just when the palette
        // input is focused), so Escape is always an escape hatch.
        if (paletteState.open) {
          closePalette();
        }
        closeReplay();
        closeBrain();
      }
    });
  }

  // ==========================================================================
  // BOOTSTRAP.
  // ==========================================================================
  function init() {
    applyStoredTheme();
    startParticles();
    wireRail();
    wirePalette();
    wireGlobalKeys();
    wireComposer();
    wireDossier();
    wireGlobe();
    wireVoice();
    connectWake();
    wireDragDrop();
    wireReplay();
    wireBrain();
    wireArena();
    wireAddressClear();
    wireAgentsFilter();
    runBootSequence();

    logLine("FRIDAY HUD loaded — press Cmd/Ctrl-K");
    if (API_BASE) {
      logLine("API base: " + API_BASE);
    }

    // Models drive both the brain pill and the arena; load once up front.
    loadModels();

    // Honor a deep-link hash on first paint; default to Command.
    var initial = (window.location.hash || "").replace(/^#/, "") || "command";
    showView(initial, { fromHash: true });

    // Slow global heartbeat for the connection pill (independent of polling).
    heartbeat();
    window.setInterval(heartbeat, 15000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
