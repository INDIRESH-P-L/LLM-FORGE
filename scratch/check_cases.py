from scripts.constitutional_retrieval import get_landmark_case_chunks

cases = get_landmark_case_chunks()
print(f"Total landmark case chunks found: {len(cases)}")
for i, c in enumerate(cases):
    print(f"[{i+1}] {c.get('chunk_id')} | {c.get('title')[:60]}")
