// A one-file Cloudflare Worker that forwards FRIDAY's Discord REST calls.
//
// Why this exists: Discord's REST API sits behind Cloudflare, and Cloudflare
// rate-limits by source IP. Render's free tier puts every tenant behind one
// shared NAT, so somebody else's traffic earned a 429 with error code 1015 and a
// Retry-After of 76,255 seconds — twenty-one hours during which FRIDAY composed
// a correct answer to every message and had all of them refused at the door. She
// stayed online the whole time, because the gateway WebSocket goes to a
// different edge that was never blocked.
//
// Routing the REST calls through a Worker moves that egress onto Cloudflare's
// network instead of Render's shared address. The gateway socket still connects
// to Discord directly; only the sends come through here.
//
// Deploy (dashboard, ~1 minute):
//   1. dash.cloudflare.com -> Compute/Workers -> Create -> "Hello World" -> Deploy
//   2. Edit code, paste this file, Deploy again.
//   3. Settings -> Variables -> add a SECRET named PROXY_SECRET. Any long random
//      string; generate one with:  openssl rand -hex 24
//   4. Point FRIDAY at it:
//        FRIDAY_DISCORD_API_BASE=https://<name>.<subdomain>.workers.dev/<PROXY_SECRET>/v10
//
// The secret lives in the path rather than a header purely so that nothing else
// in FRIDAY has to change: every call site already builds URLs from one base
// string. Without it this Worker is an open relay to Discord for anyone who
// finds the hostname.

const ALLOWED = ["authorization", "content-type", "user-agent", "x-audit-log-reason"];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);
    const given = parts.shift() || "";

    // A wrong or absent secret is a 404, not a 403: an open relay should not
    // confirm to a scanner that it is a relay at all.
    if (!env.PROXY_SECRET || given !== env.PROXY_SECRET) {
      return new Response("not found", { status: 404 });
    }

    // Only ever the versioned API. Without this the secret would open a general
    // purpose fetcher pointed at anything under discord.com.
    const rest = parts.join("/");
    if (!/^v\d+\//.test(rest)) {
      return new Response("bad path", { status: 400 });
    }

    const headers = new Headers();
    for (const name of ALLOWED) {
      const value = request.headers.get(name);
      if (value) headers.set(name, value);
    }

    const init = { method: request.method, headers, redirect: "manual" };
    if (request.method !== "GET" && request.method !== "HEAD") {
      // Buffered rather than streamed: a stream body needs duplex support that
      // not every runtime along the way provides, and these payloads are small.
      init.body = await request.arrayBuffer();
    }

    let upstream;
    try {
      upstream = await fetch(`https://discord.com/api/${rest}${url.search}`, init);
    } catch (err) {
      // 502 so the caller's retry logic treats it as worth another go.
      return new Response(`proxy could not reach discord: ${err}`, { status: 502 });
    }

    // The status and the body are passed through untouched, which is the whole
    // point: FRIDAY reads Discord's own error codes and Retry-After to decide
    // whether a failure is worth retrying, and a proxy that flattened them into
    // a generic 500 would put her right back to guessing.
    const out = new Headers(upstream.headers);
    out.delete("content-encoding");
    out.delete("content-length");
    out.delete("transfer-encoding");
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: out,
    });
  },
};
