#!/usr/bin/env bash
#
# bin/setup_https.sh
# ==================
# Generate a local certificate authority and a server certificate for
# LegalMind AI, so the app can be served over https:// on the LAN.
#
# Why this is necessary: browsers only expose the microphone (Web Speech API,
# getUserMedia, MediaRecorder) in a "secure context" — an https:// origin, or
# http://localhost specifically. Serving the app over plain http:// on a LAN
# IP puts every visitor outside a secure context, and voice input fails with
# "not-allowed" no matter what the user clicks. See docs/https_and_voice.md.
#
# Let's Encrypt is not usable on this host: the server sits on a private
# RFC1918 address (no public DNS can point at it), ports 80/443 need root,
# and the ACME endpoint is unreachable from this network. A locally-trusted
# CA gives the same security-context guarantee without any of that.
#
# Usage:   bash bin/setup_https.sh [extra-hostname-or-IP ...]
# Output:  certs/legalmind-ca.crt   <- install this on client devices (once)
#          certs/legalmind.crt/.key <- served by uvicorn

set -euo pipefail

cd "$(dirname "$0")/.."
CERT_DIR="certs"
DAYS_CA=3650
DAYS_LEAF=825          # browsers reject leaf certs valid for much longer
mkdir -p "$CERT_DIR"

# ── Collect every name the app will be reached by ────────────────────────
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
mapfile -t HOST_IPS < <(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | grep -v '^127\.' || true)

DNS_NAMES=("localhost" "$HOSTNAME_SHORT" "$(hostname)")
IP_NAMES=("127.0.0.1" "${HOST_IPS[@]}")

for extra in "$@"; do
    if [[ "$extra" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        IP_NAMES+=("$extra")
    else
        DNS_NAMES+=("$extra")
    fi
done

# de-duplicate while preserving order
dedupe() { printf '%s\n' "$@" | awk '!seen[$0]++'; }
mapfile -t DNS_NAMES < <(dedupe "${DNS_NAMES[@]}")
mapfile -t IP_NAMES  < <(dedupe "${IP_NAMES[@]}")

SAN=""
i=1; for d in "${DNS_NAMES[@]}"; do [ -n "$d" ] && SAN+="DNS.$i = $d"$'\n' && i=$((i+1)); done
i=1; for a in "${IP_NAMES[@]}";  do [ -n "$a" ] && SAN+="IP.$i = $a"$'\n'  && i=$((i+1)); done

echo "Certificate will be valid for:"
printf '  DNS: %s\n' "${DNS_NAMES[@]}"
printf '  IP:  %s\n' "${IP_NAMES[@]}"
echo

# ── 1. Local CA (only if it does not already exist) ──────────────────────
# Regenerating the CA would invalidate it on every device that installed it,
# so this step is deliberately idempotent.
if [ ! -f "$CERT_DIR/legalmind-ca.crt" ]; then
    echo "→ creating local CA…"
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$CERT_DIR/legalmind-ca.key" \
        -out    "$CERT_DIR/legalmind-ca.crt" \
        -days   "$DAYS_CA" -sha256 \
        -subj   "/C=IN/O=LegalMind AI/CN=LegalMind AI Local CA" \
        -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
        -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null
    chmod 600 "$CERT_DIR/legalmind-ca.key"
else
    echo "→ reusing existing CA (devices that trust it stay working)"
fi

# ── 2. Server certificate, signed by that CA ─────────────────────────────
echo "→ issuing server certificate…"
cat > "$CERT_DIR/openssl.cnf" <<EOF
[req]
distinguished_name = dn
req_extensions     = ext
prompt             = no

[dn]
C  = IN
O  = LegalMind AI
CN = ${DNS_NAMES[1]:-localhost}

[ext]
subjectAltName         = @alt
basicConstraints       = CA:FALSE
keyUsage               = critical,digitalSignature,keyEncipherment
extendedKeyUsage       = serverAuth

[alt]
$SAN
EOF

openssl req -newkey rsa:2048 -nodes \
    -keyout "$CERT_DIR/legalmind.key" \
    -out    "$CERT_DIR/legalmind.csr" \
    -config "$CERT_DIR/openssl.cnf" 2>/dev/null

openssl x509 -req \
    -in "$CERT_DIR/legalmind.csr" \
    -CA "$CERT_DIR/legalmind-ca.crt" -CAkey "$CERT_DIR/legalmind-ca.key" \
    -CAcreateserial \
    -out "$CERT_DIR/legalmind.crt" \
    -days "$DAYS_LEAF" -sha256 \
    -extfile "$CERT_DIR/openssl.cnf" -extensions ext 2>/dev/null

chmod 600 "$CERT_DIR/legalmind.key"
rm -f "$CERT_DIR/legalmind.csr"

echo
echo "✅ certificates written to $CERT_DIR/"
openssl x509 -in "$CERT_DIR/legalmind.crt" -noout -dates -ext subjectAltName | sed 's/^/   /'
echo
echo "Next:  bash bin/serve_https.sh"
