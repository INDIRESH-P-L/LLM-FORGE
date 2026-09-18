from datasets import load_dataset
import json
import os

OUTPUT = os.path.expanduser(
    "~/LegalMindAI/data/raw/legislation/selected_acts.jsonl"
)

TARGETS = [
    "Constitution of India",
    "Bharatiya Nyaya Sanhita",
    "Bharatiya Nagarik Suraksha Sanhita",
    "Bharatiya Sakshya Adhiniyam",
    "Indian Contract Act",
    "Specific Relief Act",
    "Transfer of Property Act",
    "Code of Civil Procedure",
    "Consumer Protection Act",
    "Information Technology Act",
    "Right to Information Act",
    "Arbitration and Conciliation Act",
    "Companies Act",
    "Negotiable Instruments Act",
    "Protection of Women from Domestic Violence Act",
    "Juvenile Justice",
    "POCSO",
    "Motor Vehicles Act",
    "Limitation Act",
    "Legal Services Authorities Act",
]

def match_title(title):
    title = title.lower()
    return [x for x in TARGETS if x.lower() in title]

print("Starting Open India Law streaming...")
print("This will NOT download the complete dataset.")

ds = load_dataset(
    "vaquill/open-india-law",
    "legislation",
    split="train",
    streaming=True
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

counts = {}
rows = 0

with open(OUTPUT, "w", encoding="utf-8") as out:

    for row in ds:
        rows += 1

        title = row.get("title") or ""
        matches = match_title(title)

        if not matches:
            continue

        act = matches[0]
        counts[act] = counts.get(act, 0) + 1

        record = {
            "act_name": act,
            "act_id": row.get("act_id"),
            "chunk_id": row.get("chunk_id"),
            "title": title,
            "chapter": row.get("chapter"),
            "section_number": row.get("section_number"),
            "section_title": row.get("section_title"),
            "text": row.get("text"),
            "jurisdiction": row.get("jurisdiction"),
            "state": row.get("state"),
            "act_status": row.get("act_status"),
            "in_force": row.get("in_force"),
            "year": row.get("year"),
            "legal_subject": row.get("legal_subject"),
            "has_proviso": row.get("has_proviso"),
            "has_non_obstante": row.get("has_non_obstante"),
            "acts_referenced": row.get("acts_referenced"),
            "defined_terms": row.get("defined_terms"),
            "amendment_count": row.get("amendment_count"),
            "source_url": row.get("source_url"),
            "source_publisher": row.get("source_publisher"),
            "mirror_url": row.get("mirror_url"),
        }

        out.write(json.dumps(record, ensure_ascii=False) + "\n")

        if rows % 100000 == 0:
            print(
                f"Scanned {rows:,} rows | "
                f"Selected {sum(counts.values()):,}"
            )

print()
print("========== COMPLETE ==========")
print(f"Rows scanned: {rows:,}")
print(f"Rows selected: {sum(counts.values()):,}")
print(f"Output file: {OUTPUT}")
print()

for act, count in sorted(counts.items()):
    print(f"{count:8,}  {act}")

missing = [x for x in TARGETS if x not in counts]

print()
print("MISSING:")
for act in missing:
    print(" -", act)
