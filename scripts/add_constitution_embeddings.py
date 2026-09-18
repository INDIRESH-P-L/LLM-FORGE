#!/usr/bin/env python3
"""
scripts/add_constitution_embeddings.py
=======================================
LegalMind AI — Embed & Index Constitution Provisions

Encodes the newly ingested authentic Constitution provisions (Articles 13, 14, 19, 20, 21, 21A, 22, 32, 226)
with BAAI/bge-m3 and adds them to:
  - vector_db/legislation/index.faiss
  - vector_db/legislation/chunk_ids.json
  - vector_db/legislation/metadata.jsonl
  - embeddings/legislation/embeddings.npy
  - embeddings/legislation/chunk_ids.json
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("add_const_embed")


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_const = project_root / "data" / "processed" / "constitution_articles.jsonl"
    vdb_dir = project_root / "vector_db" / "legislation"
    embed_dir = project_root / "embeddings" / "legislation"

    if not processed_const.exists():
        log.error(f"Missing {processed_const}. Run scripts/ingest_constitution_articles.py first.")
        sys.exit(1)

    # 1. Read provisions
    records = []
    with open(processed_const, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    log.info(f"Loaded {len(records)} constitutional provisions to embed.")

    # 2. Check existing IDs in vector_db
    vdb_ids_file = vdb_dir / "chunk_ids.json"
    with open(vdb_ids_file, "r", encoding="utf-8") as f:
        vdb_ids = json.load(f)

    existing_set = set(vdb_ids)
    to_add = [r for r in records if r["chunk_id"] not in existing_set]
    if not to_add:
        log.info("All constitutional chunks already indexed in vector_db/legislation. Skipping.")
        return

    log.info(f"Encoding {len(to_add)} new constitutional chunks...")
    texts = [r["text"] for r in to_add]
    chunk_ids = [r["chunk_id"] for r in to_add]

    # 3. Compute embeddings
    from sentence_transformers import SentenceTransformer
    device = "cuda:4"
    try:
        model = SentenceTransformer("BAAI/bge-m3", device=device)
    except Exception as e:
        log.warning(f"Could not load on {device} ({e}), falling back to cpu")
        model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    new_embeds = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    log.info(f"Generated embeddings shape: {new_embeds.shape}")

    # 4. Add to FAISS index
    import faiss
    index_file = vdb_dir / "index.faiss"
    index = faiss.read_index(str(index_file))
    log.info(f"Current FAISS vector count: {index.ntotal}")

    index.add(new_embeds)
    log.info(f"New FAISS vector count: {index.ntotal}")
    faiss.write_index(index, str(index_file))

    # 5. Update vector_db chunk_ids.json
    vdb_ids.extend(chunk_ids)
    with open(vdb_ids_file, "w", encoding="utf-8") as f:
        json.dump(vdb_ids, f)

    # 6. Update vector_db metadata.jsonl
    vdb_meta_file = vdb_dir / "metadata.jsonl"
    with open(vdb_meta_file, "a", encoding="utf-8") as f:
        for r in to_add:
            entry = {
                "chunk_id": r["chunk_id"],
                "chunk_index": 0,
                "char_count": len(r["text"]),
                "act_name": r["act_name"],
                "document_type": r["document_type"],
                "title": r["title"],
                "article": r["article_number"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 7. Update embeddings/legislation/embeddings.npy if present
    if embed_dir.exists():
        npy_file = embed_dir / "embeddings.npy"
        embed_ids_file = embed_dir / "chunk_ids.json"
        if npy_file.exists():
            old_npy = np.load(npy_file)
            updated_npy = np.vstack([old_npy, new_embeds])
            np.save(npy_file, updated_npy)
            log.info(f"Updated embeddings.npy: {old_npy.shape} -> {updated_npy.shape}")
        if embed_ids_file.exists():
            with open(embed_ids_file, "r", encoding="utf-8") as f:
                emb_ids = json.load(f)
            emb_ids.extend(chunk_ids)
            with open(embed_ids_file, "w", encoding="utf-8") as f:
                json.dump(emb_ids, f)

    log.info(f"✅ Successfully indexed {len(to_add)} constitutional provisions in FAISS and metadata!")


if __name__ == "__main__":
    main()
