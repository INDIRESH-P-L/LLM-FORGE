#!/usr/bin/env bash
#
# bin/finish_retrieval_upgrade.sh
# ===============================
# Complete the retrieval upgrade once scripts/06_index_judgments.py finishes.
#
# Waits for the judgment index, switches the app to retrieve over BOTH the
# judgments and the legislation corpora, proves a judgment is now retrievable,
# re-runs the legal evaluation, and re-fits the confidence thresholds against
# the new results.
#
# Safe to run while the index is still building — it waits.
#
#   setsid nohup bash bin/finish_retrieval_upgrade.sh \
#        > logs/finish_upgrade.log 2>&1 < /dev/null &

set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IDX=vector_db/judgments/index.faiss
CA=certs/legalmind-ca.crt
BASE=https://localhost:8443

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ── 1. wait for the embedding job ────────────────────────────────────────
say "waiting for the judgment index…"
while pgrep -f "06[_]index_judgments" >/dev/null 2>&1; do
    line=$(grep -E "shard " logs/index_judgments.log 2>/dev/null | tail -1)
    [ -n "$line" ] && say "  ${line#*INFO }"
    sleep 120
done

if [ ! -f "$IDX" ]; then
    say "FAILED: the index job exited without producing $IDX"
    tail -20 logs/index_judgments.log
    exit 1
fi
say "index present: $(du -h "$IDX" | cut -f1)"
$PY - <<'EOF'
import json, pathlib
cfg = json.loads(pathlib.Path("vector_db/judgments/index_config.json").read_text())
print(f"  vectors={cfg['num_vectors']:,} dim={cfg['embedding_dim']} type={cfg['index_type']}")
print(f"  text quality: {cfg.get('text_quality_counts')}")
EOF

# ── 2. restart the app over both corpora ─────────────────────────────────
say "restarting the app with judgments + legislation retrieval…"
for pid in $(pgrep -f "app[.]main:app" 2>/dev/null); do kill -INT "$pid" 2>/dev/null || true; done
sleep 8

export LEGALMIND_RETRIEVAL_SETS="data/chunks/judgments/|vector_db/judgments/,data/chunks/legislation/|vector_db/legislation/"
setsid nohup bash bin/serve_https.sh > logs/uvicorn-https.log 2>&1 < /dev/null &
disown 2>/dev/null || true

for _ in $(seq 1 90); do
    curl -sk -m 5 "$BASE/health" >/dev/null 2>&1 && break
    sleep 5
done
curl -sk -m 5 "$BASE/health" >/dev/null 2>&1 || { say "FAILED: app did not come up"; exit 1; }
say "app is up"

# ── 3. prove a judgment is now retrievable ───────────────────────────────
say "probe: Kesavananda Bharati (returned ZERO judgments before this change)"
curl -sk -m 900 -X POST "$BASE/api/chat" -H 'Content-Type: application/json' \
  -d '{"query":"What did the Supreme Court hold in Kesavananda Bharati v State of Kerala?","client_token":"post-index-probe"}' \
  | $PY -c "
import json,sys
d=json.load(sys.stdin); a=d['assistant_message']; md=a['metadata']
cites=a['citations'] or []
cases=[c for c in cites if (c.get('document_type') or '')=='case_law']
print(f'  retrieved={len(cites)}  judgments={len(cases)}')
for c in cases[:5]:
    print(f\"    {str(c.get('case_name'))[:56]:58} {c.get('citation')}\")
v=md.get('verification') or {}
print(f\"  verified={v.get('grounded')}/{v.get('adjudicable')} fabricated={v.get('unverified')} support={md.get('support')}\")
if not cases: print('  WARNING: still no judgments retrieved')
"

# ── 4. re-run the evaluation ─────────────────────────────────────────────
say "running the legal evaluation (after_retrieval)…"
$PY evaluation/run_legal_eval.py --label after_retrieval 2>&1 | tail -45

# ── 5. re-fit confidence thresholds ──────────────────────────────────────
say "re-fitting confidence thresholds…"
$PY evaluation/calibrate_confidence.py evaluation/legal_eval_after_retrieval.json
rc=$?
if [ $rc -eq 0 ]; then
    say "the support score now separates correct from wrong answers."
    say "ACTION: set THRESHOLDS in app/services/answer_grounding.py to the fitted values,"
    say "        then export LEGALMIND_CONFIDENCE_CALIBRATED=1 and restart to publish labels."
else
    say "the support score still does not separate correct from wrong answers;"
    say "confidence labels stay suppressed (this is the intended behaviour)."
fi

# ── 6. before / after ────────────────────────────────────────────────────
say "before / after:"
$PY - <<'EOF'
import json, pathlib
def load(p):
    f = pathlib.Path(p)
    return json.loads(f.read_text())["summary"] if f.exists() else None
rows = [("BEFORE (legislation only)", load("evaluation/legal_eval_before.json")),
        ("AFTER  (trust layer)",      load("evaluation/legal_eval_after_fixed.json")),
        ("AFTER  (+ judgments)",      load("evaluation/legal_eval_after_retrieval.json"))]
rows = [(n, s) for n, s in rows if s]
hdr = f"{'metric':<34}" + "".join(f"{n[:22]:>24}" for n, _ in rows)
print(hdr); print("-"*len(hdr))
def line(label, fn):
    print(f"{label:<34}" + "".join(f"{fn(s):>24}" for _, s in rows))
line("overall accuracy",   lambda s: f"{s['accuracy']*100:.1f}%")
for t in ("citation_lookup","case_metadata","nonexistent_case","general_legal"):
    line("  "+t, lambda s,t=t: f"{s['by_type'].get(t,{}).get('accuracy',0)*100:.1f}%")
line("citation halluc. rate", lambda s: f"{s['citations'].get('citation_hallucination_rate',0)*100:.1f}%")
line("answers w/ bad authority", lambda s: f"{s['citations'].get('pct_answers_affected',0):.1f}%")
line("avg chunks retrieved", lambda s: f"{s.get('avg_retrieved_chunks',0):.1f}")
EOF
say "done."
