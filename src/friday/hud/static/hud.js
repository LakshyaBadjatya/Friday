/*
  FRIDAY HUD — Command Centre cockpit controller (no-build, vanilla ES2022).

  A live, single-page cockpit over the EXISTING same-origin FRIDAY endpoints. No
  bundler, no framework, no CDN: every line here is plain ES2022 the browser runs
  directly. The page NEVER eval()s server output — endpoint data is rendered as
  TEXT nodes only, so a compromised or surprising payload can paint text but can
  never execute.

  Systems, each consuming an existing endpoint and degrading gracefully if it is
  off (404 / disabled / unreachable):

    1. ROSTER       GET /roster        — a card per persona; a card lights up when
                                         a recent trace used its mode (poll
                                         /admin/traces). Click to address it.
    2. TRACE FLOW   GET /admin/traces  — recent turns as animated route->dispatch
       + REPLAY     GET /admin/audit     ->synth flows; click to replay spans +
                                         matching audit rows.
    3. APPROVALS    POST /chat         — when a reply needs confirmation, a rich
                                         card with Approve/Deny re-sends the turn
                                         with confirmed:true.
    4. DOSSIER      POST /chat         — a search box that asks FRIDAY about a
                                         person/project and renders the dossier.
    5. DRAG-DROP    POST /rag/ingest   — drop a file onto the HUD to ingest it;
       RAG                               afterwards it's askable via /chat.
    6. GLOBE        GET /maps          — a button that opens the 3D globe.
    7. VOICE        Web Speech + /chat — a mic that dictates a turn to FRIDAY.
    8. METRICS      GET /admin/metrics — telemetry, per-mode counts, and an audit
       + AUDIT      GET /admin/audit     panel with a verify badge from
                    GET /admin/audit/verify.

  Plus the arc-reactor BOOT SEQUENCE and the Cmd/Ctrl-K command palette retained
  from the original cockpit.

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
    while (log.childNodes.length > 40) {
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
  // Same-origin by default. ?api=<url> or window.FRIDAY_API_BASE overrides it so
  // a locally-served HUD can drive a remote FRIDAY. A trailing slash is trimmed.
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
  // Auth (if the gateway requires it) is the browser/gateway's concern; the HUD
  // ships no credentials of its own.

  /** GET a JSON endpoint; resolves to parsed JSON or throws. */
  async function getJSON(path) {
    var resp = await fetch(url(path), {
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
    var resp = await fetch(url(path), {
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

  // Who the next free-form question is addressed to (a persona name or null).
  var addressTarget = null;

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

  // --- arc-reactor boot sequence ---------------------------------------------
  function runBootSequence() {
    var boot = el("boot");
    var text = el("boot-text");
    var bar = el("boot-bar");
    var steps = [
      "initializing reactor…",
      "spinning up containment rings…",
      "mustering persona roster…",
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

  // --- chat helper (shared by dossier, voice, palette, approvals) -----------
  // The HTTP /chat contract is {session_id, text} -> {text, mode, route, audio}.
  // We additionally send confirmed:true on an approval re-send; the backend
  // ignores unknown body fields, and a confirming follow-up turn proceeds.

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
    return postJSON("/chat", body);
  }

  // A reply "needs confirmation" when the backend says so explicitly
  // (data.needs_confirmation), or — since the documented /chat body is
  // {text, mode, route, audio} with no such field — when the in-character reply
  // is the orchestrator's confirm question. We match the stable confirm phrasing
  // ("confirm before I act" / "Reply to confirm") rather than guessing.
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

  // --- ROSTER panel ----------------------------------------------------------
  // GET /roster lists FRIDAY + 8 specialists. Each card lights up when a recent
  // trace's mode maps to that persona; clicking a card addresses that persona.
  var rosterByName = {};
  // Coarse trace-mode -> persona mapping, mirroring the roster's intent map so a
  // recent trace lights up the persona that would have handled it.
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

  /** Resolve a trace mode string to a persona name (FRIDAY when unknown). */
  function personaForMode(mode) {
    if (!mode) {
      return "FRIDAY";
    }
    var key = String(mode).toLowerCase();
    if (MODE_TO_PERSONA[key]) {
      return MODE_TO_PERSONA[key];
    }
    // Substring fallback: e.g. "finance_query" -> GECKO.
    for (var token in MODE_TO_PERSONA) {
      if (Object.prototype.hasOwnProperty.call(MODE_TO_PERSONA, token)) {
        if (key.indexOf(token) !== -1) {
          return MODE_TO_PERSONA[token];
        }
      }
    }
    return "FRIDAY";
  }

  function setAddress(name) {
    addressTarget = name && name !== "FRIDAY" ? name : null;
    var bar = el("address-bar");
    var target = el("address-target");
    if (target) {
      target.textContent = name || "FRIDAY";
    }
    if (bar) {
      bar.hidden = !addressTarget;
    }
    // Reflect selection on the roster cards.
    var grid = el("roster-grid");
    if (grid) {
      var cards = grid.querySelectorAll(".persona");
      for (var i = 0; i < cards.length; i++) {
        cards[i].classList.toggle(
          "is-selected",
          cards[i].getAttribute("data-name") === name
        );
      }
    }
    if (addressTarget) {
      toast("Addressing " + addressTarget + " — type your message");
    }
  }

  async function loadRoster() {
    var grid = el("roster-grid");
    if (!grid) {
      return;
    }
    var data;
    try {
      data = await getJSON("/roster");
    } catch (err) {
      // Roster is always-available server-side; a failure means offline.
      clear(grid);
      grid.appendChild(mk("div", "panel-empty", "roster unavailable"));
      return;
    }
    var personas = (data && data.personas) || [];
    var count = el("roster-count");
    if (count) {
      count.textContent = personas.length ? String(personas.length) : "";
    }
    rosterByName = {};
    clear(grid);
    personas.forEach(function (p) {
      rosterByName[p.name] = p;
      var card = mk("button", "persona");
      card.type = "button";
      card.setAttribute("data-name", p.name);
      card.title = (p.title || "") + " — click to address";
      var dot = mk("span", "persona-dot");
      var body = mk("span", "persona-body");
      body.appendChild(mk("span", "persona-name", p.name));
      body.appendChild(mk("span", "persona-title", p.title || ""));
      var scope = (p.scope || []).slice(0, 4).join(" · ");
      if ((p.scope || []).length > 4) {
        scope += " · …";
      }
      body.appendChild(mk("span", "persona-scope", scope));
      card.appendChild(dot);
      card.appendChild(body);
      card.addEventListener("click", function () {
        // Toggle: clicking the addressed persona clears the address.
        setAddress(addressTarget === p.name ? null : p.name);
      });
      grid.appendChild(card);
    });
    logLine("roster: " + personas.length + " personas", "ok");
  }

  /** Light up persona cards that a recent trace used; fade the rest. */
  function lightRoster(traces) {
    var grid = el("roster-grid");
    if (!grid) {
      return;
    }
    // Mark a persona "warm" when any of the most-recent traces resolves to it.
    // /admin/traces returns the recent window already (oldest-first), so we take
    // the newest few as the active set; a longer-idle persona simply fades out.
    var active = {};
    (traces || []).slice(-6).forEach(function (t) {
      active[personaForMode(t.mode)] = true;
    });
    var cards = grid.querySelectorAll(".persona");
    for (var i = 0; i < cards.length; i++) {
      var name = cards[i].getAttribute("data-name");
      cards[i].classList.toggle("is-active", !!active[name]);
    }
  }

  // --- TRACE FLOW + REPLAY ---------------------------------------------------
  // Render recent /admin/traces as route->dispatch->synth flows. Clicking a flow
  // replays its spans and the matching /admin/audit rows (same correlation id).
  var lastTraces = [];
  var lastAudit = { tool_calls: [], security: [] };

  function spanDurationMs(span) {
    if (span && typeof span.start === "number" && typeof span.end === "number") {
      var ms = (span.end - span.start) * 1000;
      return ms >= 0 ? ms : 0;
    }
    return null;
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
    // Newest first for readability.
    var traces = lastTraces.slice().reverse();
    traces.forEach(function (t) {
      var row = mk("button", "flow");
      row.type = "button";
      row.title = "Replay " + t.correlation_id;
      var head = mk("span", "flow-head");
      head.appendChild(mk("span", "flow-mode", t.mode || "—"));
      head.appendChild(
        mk("span", "flow-cid", String(t.correlation_id).slice(0, 8))
      );
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
        // Stagger the entrance so the pipe "flows" left-to-right.
        step.style.animationDelay = idx * 90 + "ms";
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
    var box = el("replay");
    var idNode = el("replay-id");
    var spansNode = el("replay-spans");
    var auditNode = el("replay-audit");
    if (!box || !spansNode || !auditNode) {
      return;
    }
    box.hidden = false;
    if (idNode) {
      idNode.textContent = String(trace.correlation_id).slice(0, 12);
    }
    // Replay the spans with a staggered reveal so it "plays back".
    clear(spansNode);
    (trace.spans || []).forEach(function (s, idx) {
      var item = mk("div", "rspan");
      item.style.animationDelay = idx * 140 + "ms";
      item.appendChild(mk("span", "rspan-name", s.name || "span"));
      var ms = spanDurationMs(s);
      item.appendChild(
        mk("span", "rspan-ms", ms != null ? Math.round(ms) + "ms" : "open")
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
    // Matching audit rows share the correlation id.
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

  function wireReplayClose() {
    var btn = el("replay-close");
    if (btn) {
      btn.addEventListener("click", function () {
        var box = el("replay");
        if (box) {
          box.hidden = true;
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
      lightRoster(lastTraces);
    } catch (err) {
      // Traces are flagged behind the tracer; quietly skip when absent.
      var tag2 = el("flow-tag");
      if (tag2) {
        tag2.textContent = "offline";
      }
    }
  }

  // --- AUDIT panel + verify badge -------------------------------------------
  function renderAudit() {
    var list = el("audit-list");
    if (!list) {
      return;
    }
    clear(list);
    var rows = (lastAudit.tool_calls || []).slice(-8).reverse();
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
      /* audit best-effort; metrics drives the pill. */
    }
    // Verify badge — independent of the audit list (separate endpoint).
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

  // --- METRICS / telemetry ---------------------------------------------------
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

  // --- DOSSIER cards ---------------------------------------------------------
  // A search box that asks FRIDAY about a person/project; the reply renders as a
  // dossier card. Goes through /chat (same as any question), addressed to the
  // currently-selected persona if one is active.
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
      // A dossier reply can itself be a confirm question; surface it.
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
      setOnline(false);
    }
  }

  function wireDossier() {
    var form = el("dossier-form");
    var input = el("dossier-input");
    var go = el("dossier-go");
    if (!form || !input) {
      return;
    }
    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      runDossier(input.value);
      input.value = "";
    });
    if (go) {
      go.addEventListener("click", function () {
        runDossier(input.value);
        input.value = "";
      });
    }
  }

  // --- APPROVAL cards --------------------------------------------------------
  // When a /chat reply needs confirmation, show a rich card with the pending
  // action + Approve/Deny. Approve re-sends the SAME turn with confirmed:true.
  function maybeApproval(data, originalText) {
    if (!replyNeedsConfirmation(data)) {
      return false;
    }
    var panel = el("approvals-panel");
    var cards = el("approval-cards");
    if (!panel || !cards) {
      return false;
    }
    panel.hidden = false;

    var card = mk("div", "card approval");
    var head = mk("div", "card-head");
    head.appendChild(mk("span", "card-title", "Confirmation required"));
    head.appendChild(mk("span", "card-meta", (data && data.mode) || "action"));
    card.appendChild(head);

    card.appendChild(mk("div", "card-body", (data && data.text) || ""));

    // Surface any structured tool/args the backend included (best-effort: the
    // documented body has none, but we render them when present).
    var detail = data && (data.confirmation || data.pending || data.tool);
    if (detail && typeof detail === "object") {
      var pre = mk("div", "card-detail");
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

    var actions = mk("div", "card-actions");
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
        var body = card.querySelector(".card-body");
        if (body) {
          body.textContent = (confirmed && confirmed.text) || "Done.";
        }
        var meta = card.querySelector(".card-meta");
        if (meta) {
          meta.textContent = "approved";
        }
        logLine("approved: " + originalText, "ok");
        // If the confirmed turn ALSO needs confirmation (chained), re-card it.
        maybeApproval(confirmed, originalText);
      } catch (err) {
        approve.disabled = false;
        deny.disabled = false;
        toast("Approve failed: " + (err && err.message ? err.message : err));
      }
    });

    deny.addEventListener("click", function () {
      card.classList.add("denied");
      var meta = card.querySelector(".card-meta");
      if (meta) {
        meta.textContent = "denied";
      }
      approve.disabled = true;
      deny.disabled = true;
      logLine("denied: " + originalText);
    });

    actions.appendChild(approve);
    actions.appendChild(deny);
    card.appendChild(actions);

    cards.insertBefore(card, cards.firstChild);
    while (cards.childNodes.length > 5) {
      cards.removeChild(cards.lastChild);
    }
    toast("FRIDAY needs your confirmation");
    return true;
  }

  // --- DRAG-DROP RAG ingest --------------------------------------------------
  // Dropping a file POSTs it to /rag/ingest as multipart; afterwards it is
  // askable through the normal /chat path. 404 (RAG disabled) is reported, never
  // a crash.
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
    } catch (err) {
      toast("Ingest failed: " + (err && err.message ? err.message : err));
      logLine("ingest failed: " + file.name, "err");
    }
  }

  function wireDragDrop() {
    var overlay = el("drop-overlay");
    var depth = 0;

    window.addEventListener("dragenter", function (ev) {
      if (ev.dataTransfer && Array.prototype.indexOf.call(ev.dataTransfer.types || [], "Files") !== -1) {
        ev.preventDefault();
        depth++;
        if (overlay) {
          overlay.classList.add("is-active");
        }
      }
    });
    window.addEventListener("dragover", function (ev) {
      if (ev.dataTransfer && Array.prototype.indexOf.call(ev.dataTransfer.types || [], "Files") !== -1) {
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

  // --- GLOBE button ----------------------------------------------------------
  function wireGlobe() {
    var btn = el("btn-globe");
    if (btn) {
      btn.addEventListener("click", function () {
        // Open the existing /maps surface (honors the configured API base).
        window.open(url("/maps"), "_blank", "noopener");
        logLine("opened /maps");
      });
    }
  }

  // --- VOICE (Web Speech) ----------------------------------------------------
  // A mic that dictates one turn to FRIDAY. Falls back to a friendly toast when
  // SpeechRecognition is unavailable (most non-Chromium browsers).
  function wireVoice() {
    var btn = el("btn-mic");
    if (!btn) {
      return;
    }
    var Rec =
      window.SpeechRecognition || window.webkitSpeechRecognition || null;
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
      toast("Heard: " + said);
      logLine("voice: " + said);
      sendChat(withAddress(said))
        .then(function (data) {
          toast("FRIDAY: " + String((data && data.text) || "").slice(0, 90));
          maybeApproval(data, withAddress(said));
        })
        .catch(function (err) {
          toast("Voice chat failed: " + (err && err.message ? err.message : err));
        });
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

  function wireAddressClear() {
    var btn = el("address-clear");
    if (btn) {
      btn.addEventListener("click", function () {
        setAddress(null);
      });
    }
  }

  // --- command palette -------------------------------------------------------
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
        var data = await sendChat(withAddress(q));
        logLine("chat: " + q, "ok");
        maybeApproval(data, withAddress(q));
        return String(data && data.text ? data.text : "(no reply)");
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
        runDossier(q);
        return "Dossier requested for: " + q;
      },
    },
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
      // The free-query path passes the typed text to argument-taking commands
      // (ask/dossier); fixed commands ignore it.
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
      }
    });
  }

  // --- bootstrap -------------------------------------------------------------
  function init() {
    startParticles();
    wirePalette();
    wireGlobalKeys();
    wireDossier();
    wireGlobe();
    wireVoice();
    wireDragDrop();
    wireReplayClose();
    wireAddressClear();
    runBootSequence();
    logLine("Command Centre loaded — press Cmd/Ctrl-K");
    if (API_BASE) {
      logLine("API base: " + API_BASE);
    }

    // Initial + periodic data pulls. Each is independent and failure-isolated:
    // a dead endpoint flips its own panel to "offline" but never throws.
    loadRoster();
    refreshTelemetry();
    refreshTraces();
    refreshAudit();
    window.setInterval(refreshTelemetry, 5000);
    window.setInterval(refreshTraces, 4000);
    window.setInterval(refreshAudit, 8000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
