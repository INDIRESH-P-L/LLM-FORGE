# Trustworthy legal answers: retrieval, verification, evaluation

How LegalMind AI decides whether an answer is supported, what it refuses to
say, and how to re-measure all of it.

The design premise: **on a legal tool, a confident wrong answer is worse than
no answer.** Lawyers have been sanctioned for filing AI-invented authorities.
So every component here is built to fail toward "I can't verify this" rather
than toward a fluent guess.

---

## The problem this fixes

The app had a full hybrid RAG pipeline (BM25 + FAISS/bge-m3 + cross-encoder
rerank) — but it was pointed at `vector_db/legislation/` only. Asked about
**Kesavananda Bharati**, the most cited judgment in Indian constitutional law,
it retrieved five chunks of the Constitution and two of the CPC, and **zero
judgments**. Every case citation it produced came from fine-tuned weights,
unretrieved and unchecked.

The pre-existing `vector_db/case_laws/` could not help: all 26,085 rows were
`chunk_index=0`, i.e. docket headers (`"Case title: … Parties: … Docket
number: …"`) with no legal reasoning and `citation="nan"`.

---

## Components

| file | role |
|---|---|
| `scripts/06_index_judgments.py` | build the judgment FAISS index (resumable) |
| `scripts/07_build_authority_table.py` | build the ground-truth authority DB |
| `app/services/citation_verifier.py` | check every asserted authority |
| `app/services/answer_grounding.py` | measure support; decide abstention |
| `app/services/multi_retriever.py` | retrieve across judgments **and** legislation |
| `evaluation/build_eval_set.py` | generate the eval set from the corpus |
| `evaluation/run_legal_eval.py` | score accuracy + hallucination rate |
| `evaluation/calibrate_confidence.py` | fit (or refuse to fit) confidence thresholds |
| `evaluation/audit_datasets.py` | dataset correctness audit |
| `tests/test_trust_layer.py` | 23 tests, both failure directions |

---

## 1. Retrieval

`scripts/06_index_judgments.py` indexes the Supreme Court corpus
(`in_supreme-court_judgments.parquet`, 1,113,313 chunks over 37,014 judgments).

Every chunk is tagged with **`script`** and **`text_quality`**
(`clean` / `degraded` / `corrupt`), because ~59% of the corpus is machine
translation damaged by legacy-font extraction — lost word boundaries, dropped
Tamil viramas, Latin glyphs wedged into Gurmukhi words. Measured on the live
run: **41% clean, 29% degraded, 30% corrupt.**

That text stays searchable (it is legitimate for regional-language queries) but
`answer_grounding` refuses to count `corrupt` passages as support, so garbled
text can never be quoted as authority.

The run is **resumable** — embeddings flush to shards, so an interrupted job
continues rather than repeating six GPU-hours.

```bash
.venv/bin/python scripts/06_index_judgments.py            # full build
.venv/bin/python scripts/06_index_judgments.py --limit 3000   # smoke test
```

### Retrieving over both corpora

A legal answer usually needs the statute *and* the judgment construing it. The
app could previously load one index; `LEGALMIND_RETRIEVAL_SETS` takes a
comma-separated list of `<chunks_dir>|<vector_db_dir>` pairs:

```bash
export LEGALMIND_RETRIEVAL_SETS="data/chunks/judgments/|vector_db/judgments/,data/chunks/legislation/|vector_db/legislation/"
```

`MultiRetriever` merges by reranker score, falls back to RRF, dedupes, and
renumbers `[Source N]` so prompt citations match the returned list.
`scripts/retriever.py` is untouched.

---

## 2. Citation verification

`data/verification/authorities.db` is the ground truth — **37,014 judgments**
(title, reporter citation, court, decision date, judges) and **206,737
statutory provisions**, built by `scripts/07_build_authority_table.py` from the
judgment parquet, the 37 legislation parquets (which carry a structured
`section_number` column), and the official Government Constitution text.

It is deliberately independent of FAISS: verification must work even when
retrieval returns nothing, and must not depend on a six-hour embedding job.

### Four verdicts

| verdict | meaning |
|---|---|
| `grounded` | appears in the passages retrieved **for this answer** — strongest |
| `in_corpus` | exists in our data but was not retrieved; real, but the claim attached to it is not evidenced |
| `source_gap` | **our corpus cannot check this either way** |
| `unverified` | not found. Treat as fabricated |

`source_gap` is the one that stops the verifier lying in the other direction.
Our judgments carry **only S.C.R. citations** — zero S.C.C. or A.I.R. Indian
practice usually cites S.C.C., so `(1973) 4 SCC 225` (Kesavananda's real SCC
citation) is unverifiable *by construction*. Reporting that as fabricated would
tell a user a genuine authority is fake — a different but equally serious
failure. The hallucination rate is computed over **adjudicable** authorities
only, so it measures the model, not our coverage gaps.

**Measured verifier accuracy: 100% recall on 8 landmark cases, 0 false accepts
on 5 fabricated ones.**

### Three bugs worth knowing about (all now regression-tested)

1. Case-name capture ran into the following verb (`"State of Kerala establi"`),
   so Kesavananda looked fabricated. Fixed by anchoring on capitalisation.
2. `LIMIT 1` returned an arbitrary match — `"Maneka Gandhi v. Union of India"`
   matched *Buffalo Traders v. Maneka Gandhi*, scored 0.25, and was rejected.
   Now all candidates are scored and the best is taken.
3. Constitution articles were extracted with a heading pattern that yielded 521
   "articles" including invented ones like `12AA`. Now only explicit mentions
   are accepted (198 articles) — what we miss becomes `source_gap`, never a
   false pass.

---

## 3. Grounding and abstention

`answer_grounding.assess()` produces a 0–1 **support** score from things that
can be measured about the specific answer:

- reranker strength of the retrieved set (logistic-mapped)
- share of asserted authorities that are grounded / in corpus / unverifiable
- whether an Article/Section named in the *question* actually appears in the
  retrieved text
- source quality (`corrupt` passages excluded)

Each unverifiable authority applies a **non-linear penalty** — one fabricated
citation is enough to make an answer unusable.

Below `LEGALMIND_ABSTAIN_FLOOR` (default 0.22) the answer is replaced with an
explicit refusal naming the reason and telling the user to consult a
professional. The withheld text is kept in metadata for debugging, never shown.

> **A caution from building this.** The first version abstained on *everything*
> and reported 0% hallucination — because `_usable()` read `text`/`excerpt`
> while `run_rag` emits `text_preview`, so every passage was discarded. General
> legal accuracy went 100% → 0%. A tool that refuses everything has a perfect
> hallucination rate and no value. **Always read the accuracy number next to
> the hallucination number.**

---

## 4. Confidence is suppressed until it is earned

`evaluation/calibrate_confidence.py` reports separation, AUC and ECE, and
grid-searches thresholds that keep buckets monotone.

On the current data it returns:

```
separation : +0.015
AUC        : 0.629   (0.5 = no signal)
ECE        : 0.266
VERDICT: the support score does not separate correct from wrong answers.
```

So **no confidence label is published.** `CALIBRATED` in
`answer_grounding.py` (or `LEGALMIND_CONFIDENCE_CALIBRATED=1`) gates it, and
the API/UI show the evidence instead:

> `16 of 29 authorities verified against retrieved sources; 8 usable source passage(s)`

The score is uninformative today for a structural reason: 70% of the eval fails
because judgments are not yet retrievable, which the support score cannot see —
legislation chunks retrieve *confidently* for judgment questions. Re-fit after
the judgment index is live; only turn the label on if the fitter reports real
separation.

---

## 5. Evaluation

`evaluation/legal_eval_set.jsonl` — 40 questions, ground truth drawn from the
same corpus the verifier checks, so there is no separate answer key to drift.

| type | n | correct behaviour |
|---|---|---|
| `citation_lookup` | 14 | produce the reporter citation |
| `case_metadata` | 14 | court + decision date |
| `nonexistent_case` | 6 | **refuse** — the case does not exist |
| `general_legal` | 6 | known key terms present |

```bash
.venv/bin/python evaluation/build_eval_set.py --n 40
.venv/bin/python evaluation/run_legal_eval.py --label after_retrieval
.venv/bin/python evaluation/calibrate_confidence.py evaluation/legal_eval_after_retrieval.json
```

### Results so far

| metric | before | after (trust layer) |
|---|---|---|
| overall accuracy | 15.0% | **30.0%** |
| non-existent cases refused | **0%** | **100%** |
| general legal | 100% | 100% |
| citation lookup / metadata | 0% | 0% ← needs the judgment index |
| **citation hallucination rate** | **5.7%** | **0.7%** |
| **case-name fabrication** | **9.7%** | **1.5%** |
| answers with an unverifiable authority | 62.5% | 5.0% |

`citation_lookup` and `case_metadata` stay at 0% until judgments are
retrievable. That is the gating item, not the trust layer.

---

## 6. Finishing the upgrade

`bin/finish_retrieval_upgrade.sh` waits for the index job, switches retrieval
to judgments + legislation, re-runs the Kesavananda probe, re-runs the eval as
`after_retrieval`, re-fits thresholds, and prints a three-way comparison.

```bash
setsid nohup bash bin/finish_retrieval_upgrade.sh > logs/finish_upgrade.log 2>&1 < /dev/null &
```

---

## Known limits

- **No S.C.C./A.I.R. citations** in the corpus, so those are `source_gap`, not
  verified. A parallel-citation table would close this.
- **198 of ~395 Constitution articles.** The official text uses bare headings
  (`141. Law declared by Supreme Court…`), which the conservative extractor
  does not accept. Missing ones degrade to `source_gap`.
- **Confidence is not calibrated.** Stated plainly rather than papered over.
- **Supreme Court only.** High Court parquets are indexed nowhere; their
  `citation` column is 100% null anyway.
- **Holdings are not verified, only authorities.** The verifier confirms a case
  exists and was retrieved — not that it says what the answer claims.
- **~11% duplicate rows** in the SFT train and eval splits (train/eval leakage
  itself is clean: 0.25% exact, 0 near-duplicates).
