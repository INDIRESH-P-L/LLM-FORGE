# HTTPS and voice input

Why voice input stopped working when the app moved off `localhost`, what was
done about it, and how to run the server now.

---

## The problem

Voice typing worked on `localhost` and fails on the shared server with
**"Voice input error: not-allowed"**.

## Root cause

Voice input is the **Web Speech API** (`SpeechRecognition` /
`webkitSpeechRecognition`), in `app/static/app.js`. Browsers expose speech
recognition and microphone capture only inside a **secure context**:

| Origin | Secure context? | Mic works? |
|---|---|---|
| `https://anything` | yes | yes |
| `http://localhost`, `http://127.0.0.1` | yes (special-cased) | yes |
| `http://192.168.4.99:8080` | **no** | **no** |
| `http://any-hostname` | **no** | **no** |

`localhost` is treated as "potentially trustworthy" regardless of scheme,
which is exactly why this worked during development and broke on deployment.
On a non-secure origin the failure is unconditional: the browser fires
`SpeechRecognitionErrorEvent` with `error = "not-allowed"` **before** showing
a permission prompt, so clicking "Allow" changes nothing. The old code
surfaced that raw code verbatim — hence the bare "not-allowed" message.

### Ruled out

- **Missing `allow="microphone"` on an iframe.** There are no iframes: the app
  is FastAPI serving `app/static/` directly (`grep -rn iframe app/` finds
  nothing). It is not Streamlit or Gradio — neither is installed — so there is
  no component iframe to add a permissions attribute to.
- **A `getUserMedia`/`MediaRecorder` fallback to the server-side Whisper
  endpoint** (`app/api/speech.py`, `POST /api/speech-to-text`). That would not
  help: `getUserMedia` is gated by the *same* secure-context rule. There is no
  in-page workaround for this. HTTPS is the only fix.

---

## What was done

### 1. HTTPS with a locally-trusted certificate

Let's Encrypt is not possible on this host, for four independent reasons:

- the server is on a **private address** (`192.168.4.99`) — no public DNS
  record can point at it, so neither HTTP-01 nor TLS-ALPN-01 can validate;
- **ports 80 and 443 need root**, and this account does not have it
  (`sudo` requires a password);
- **the ACME endpoint is unreachable** from this network;
- there is no domain name to put on a certificate.

Nginx/Caddy are not installed and installing them system-wide needs root. So
TLS is terminated by uvicorn itself, using a local CA:

```bash
bash bin/setup_https.sh            # once — creates certs/
bash bin/serve_https.sh            # serves https://<host>:8443
```

`bin/setup_https.sh` creates `certs/legalmind-ca.crt` (a local CA, reused on
re-runs so devices that already trust it keep working) and a server
certificate whose SANs cover `localhost`, the hostname and every LAN IP.
Port 8443 rather than 443 because low ports need root — **the secure-context
rule cares only about the scheme, not the port.**

### 2. The old HTTP URL redirects instead of serving the app

`bin/http_redirect.py` listens on `:8080` and 307-redirects everything to
`https://<same-host>:8443`, preserving path, query string and method. It is a
bare ASGI app that **loads no model** — running the full app twice would put a
second 68 GB copy of Qwen3.6-35B on the same GPU.

### 3. The error message now explains itself

`app/static/app.js` checks `window.isSecureContext` before starting
recognition. On a plain-http origin the mic button is visibly struck through,
and clicking it gives the reason plus an **"Open secure site"** button that
navigates to the HTTPS URL, instead of the cryptic `not-allowed`. Other
`SpeechRecognition` error codes (`no-speech`, `audio-capture`, `network`, …)
now map to plain-English messages too.

---

## Running the server

```bash
cd /home/sece2026-student12/LegalMindAI

# HTTPS app (this is the real one)
setsid nohup bash bin/serve_https.sh > logs/uvicorn-https.log 2>&1 < /dev/null &

# optional: bounce the old http:// link to it
setsid nohup bash bin/serve_http_redirect.sh > logs/http-redirect.log 2>&1 < /dev/null &
```

`setsid` matters: without it the process is killed when its parent shell is
reaped, which is how both listeners died mid-session during this work.

| Variable | Default | Meaning |
|---|---|---|
| `LEGALMIND_HTTPS_PORT` | `8443` | HTTPS listen port |
| `LEGALMIND_HTTP_PORT` | `8080` | redirector listen port |
| `LEGALMIND_HOST` | `0.0.0.0` | bind address |

---

## Confirming the diagnosis yourself

On the plain-http URL, open DevTools → Console and run:

```js
window.isSecureContext        // false  ← the whole problem
location.protocol             // "http:"
```

Then click the mic. The Console shows the `SpeechRecognitionErrorEvent` with
`error: "not-allowed"`, fired with no permission prompt. In
`chrome://settings/content/microphone` the site cannot even be granted access.

On `https://<host>:8443` the same two expressions give `true` and `"https:"`,
and the mic prompt appears normally.

---

## Verifying the fix

1. Open **`https://192.168.4.99:8443`** (note **https** and **8443**).
2. First visit shows a certificate warning — click **Advanced → Proceed**.
   To remove the warning entirely, install the CA once per device (below).
3. Check the padlock area shows the site as secure, or run
   `window.isSecureContext` in the console — it must be `true`.
4. Click the microphone. The browser should now ask for permission; allow it.
   Speak — the text should appear in the input box.
5. Opening the old `http://192.168.4.99:8080` should bounce you to the HTTPS
   URL automatically.

### Installing the CA (optional, removes the warning)

Copy `certs/legalmind-ca.crt` to the device, then:

- **Windows:** double-click → Install Certificate → Local Machine → *Trusted
  Root Certification Authorities*.
- **macOS:** double-click → Keychain Access → System → set to *Always Trust*.
- **Android:** Settings → Security → Encryption & credentials → Install a
  certificate → CA certificate.
- **iOS:** AirDrop/email it → Settings → Profile Downloaded → Install → then
  Settings → General → About → Certificate Trust Settings → enable it.
- **Linux/Chrome:** `chrome://settings/certificates` → Authorities → Import.

### Browser support

`SpeechRecognition` is a Chrome/Edge feature. Firefox does not implement it
and Safari's support is partial — the button already reports this rather than
failing silently. HTTPS is required in all of them regardless.

---

## If you later get a real domain

With a public DNS name and root, the standard setup is better than a local CA
(no per-device install). Point the name at the host and:

```bash
sudo apt install caddy
# /etc/caddy/Caddyfile
legalmind.example.edu {
    reverse_proxy 127.0.0.1:8443 {
        transport http { tls_insecure_skip_verify }
    }
}
sudo systemctl reload caddy
```

Caddy obtains and renews a Let's Encrypt certificate automatically. Simplest
is then to run the app over plain HTTP on `127.0.0.1:8000` and let Caddy own
TLS. A Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8443`)
also works with no root and no domain, but publishes an internal legal-research
tool to the public internet — consider whether that is acceptable first.
