#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from constitutional_retrieval import get_constitutional_retriever, is_golden_triangle_query, extract_constitutional_articles

q = (
    "Explain the differences between Article 14, Article 19, and Article 21 of the Constitution of India. "
    "Discuss how these rights are connected, their reasonable restrictions, and important Supreme Court judgments. "
    "Give the answer in a clear legal order with verified sources."
)

print(f"Is Golden Triangle: {is_golden_triangle_query(q)}")
print(f"Articles: {extract_constitutional_articles(q)}")

cr = get_constitutional_retriever()
docs, coverage = cr.retrieve(q, top_k=14)

print(f"\nRetrieved docs count: {len(docs)}")
for d in docs:
    cat = d.get("constitutional_category", "general")
    title = d.get("title", "")
    art = d.get("article_number", "")
    print(f"  [{d.get('citation_id')}] ({cat}) Art {art} — {title}")

print("\nArticle Coverage Breakdown:")
import json
print(json.dumps(coverage, indent=2))
