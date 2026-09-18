import subprocess
import sys
from pathlib import Path

repo_root = Path("/home/sece2026-student12/LegalMindAI")

# Check git status
res = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True)
lines = res.stdout.strip().split("\n") if res.stdout.strip() else []

print("=== GIT STATUS AUDIT ===")
modified_existing = []
untracked_new = []

for line in lines:
    status = line[:2]
    filename = line[3:]
    if status.startswith("?") or status.endswith("?"):
        untracked_new.append(filename)
    else:
        modified_existing.append((status, filename))

print(f"Total modified existing files: {len(modified_existing)}")
for s, f in modified_existing:
    print(f"  [{s}] {f}")

print(f"\nTotal new files: {len(untracked_new)}")
for f in untracked_new:
    print(f"  [NEW] {f}")

# Check that web UI files are not in modified_existing
web_ui_files = ["app/static", "app/index.html"]
violated = [f for s, f in modified_existing if any(w in f for w in web_ui_files)]
if violated:
    print(f"\n[VIOLATION] Web UI files modified: {violated}")
else:
    print("\n[CONFIRMED] Zero modifications to existing web UI or frontend components!")
