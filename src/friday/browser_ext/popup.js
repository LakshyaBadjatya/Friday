// © Lakshya Badjatya — Author
// FRIDAY Quick-Ask popup — POSTs to the local FRIDAY /chat endpoint and shows
// the reply. No bundler, no dependencies; the base URL is remembered via
// chrome.storage (falling back to localStorage when run outside an extension).
(function () {
  "use strict";

  var DEFAULT_BASE = "http://127.0.0.1:8000";
  var SESSION = "browser-ext";

  var q = document.getElementById("q");
  var base = document.getElementById("base");
  var send = document.getElementById("send");
  var statusEl = document.getElementById("status");
  var reply = document.getElementById("reply");

  function loadBase(cb) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(["base"], function (got) {
        cb((got && got.base) || DEFAULT_BASE);
      });
      return;
    }
    try {
      cb(window.localStorage.getItem("friday-base") || DEFAULT_BASE);
    } catch (e) {
      cb(DEFAULT_BASE);
    }
  }

  function saveBase(value) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ base: value });
      return;
    }
    try {
      window.localStorage.setItem("friday-base", value);
    } catch (e) {
      /* ignore */
    }
  }

  async function ask() {
    var text = (q.value || "").trim();
    if (!text) {
      return;
    }
    var url = (base.value || DEFAULT_BASE).replace(/\/+$/, "");
    saveBase(url);
    statusEl.textContent = "…";
    reply.textContent = "";
    try {
      var res = await fetch(url + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION, text: text }),
      });
      var data = await res.json();
      reply.textContent = data.response || data.detail || "(no reply)";
      statusEl.textContent = data.mode ? "[" + data.mode + "]" : "";
    } catch (err) {
      statusEl.textContent = "";
      reply.textContent = "Could not reach FRIDAY at " + url + " — is it running?";
    }
  }

  send.addEventListener("click", ask);
  q.addEventListener("keydown", function (ev) {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
      ask();
    }
  });

  loadBase(function (value) {
    base.value = value;
  });
})();
