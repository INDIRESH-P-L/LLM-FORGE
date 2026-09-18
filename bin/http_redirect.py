#!/usr/bin/env python3
"""
bin/http_redirect.py
====================
Tiny HTTP listener whose only job is to send browsers to the HTTPS app.

Why this exists: the microphone (Web Speech API / getUserMedia) is only
available in a secure context — https://, or http://localhost. Anyone opening
the old plain-http link lands outside one and voice input fails with
"not-allowed". Rather than run a second copy of the application on HTTP —
which would load the 35B model onto the GPU a second time — this process
loads nothing and 307-redirects every request to the HTTPS origin.

    LEGALMIND_HTTP_PORT    listen port        (default 8080)
    LEGALMIND_HTTPS_PORT   redirect target    (default 8443)
    LEGALMIND_HTTPS_HOST   target host        (default: the requested Host)

Run via bin/serve_http_redirect.sh
"""

import os

HTTPS_PORT = os.environ.get("LEGALMIND_HTTPS_PORT", "8443")
HTTPS_HOST = os.environ.get("LEGALMIND_HTTPS_HOST", "")


async def app(scope, receive, send):
    """Bare ASGI app — no FastAPI, no model, no database."""
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
        return

    if scope["type"] != "http":
        return

    # Preserve whichever host the visitor typed, so the redirect works whether
    # they came in on the LAN IP or the hostname.
    headers = {k.decode("latin-1").lower(): v.decode("latin-1")
               for k, v in scope.get("headers", [])}
    host = HTTPS_HOST or headers.get("host", "").split(":")[0] or "localhost"

    path = scope.get("path", "/")
    query = scope.get("query_string", b"").decode("latin-1")
    target = f"https://{host}:{HTTPS_PORT}{path}" + (f"?{query}" if query else "")

    # 307 keeps the method and body, so a POST that hits the old URL is not
    # silently downgraded to a GET.
    await send({
        "type": "http.response.start",
        "status": 307,
        "headers": [
            (b"location", target.encode("latin-1")),
            (b"content-type", b"text/html; charset=utf-8"),
            (b"cache-control", b"no-store"),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": (
            "<!doctype html><meta charset=utf-8>"
            f'<title>Redirecting…</title><meta http-equiv="refresh" content="0;url={target}">'
            "<style>body{font-family:system-ui;background:#0a0e1a;color:#e8ecf4;"
            "max-width:38rem;margin:18vh auto;padding:0 1.5rem;line-height:1.6}"
            "a{color:#8b9dff}</style>"
            "<h1>&#9878;&#65039; LegalMind AI has moved to HTTPS</h1>"
            f'<p>Redirecting to <a href="{target}">{target}</a>&hellip;</p>'
            "<p>Voice input needs a secure connection, so the app is served over "
            "HTTPS. Your browser may warn about the certificate the first time &mdash; "
            "choose <b>Advanced &rarr; Proceed</b>, or install "
            "<code>certs/legalmind-ca.crt</code> to remove the warning.</p>"
        ).encode("utf-8"),
    })
