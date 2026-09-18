import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fine_tuning"))
sys.path.insert(0, str(Path(__file__).parent))

print("Importing modules...", flush=True)
t0 = time.time()
from test_pipeline import TestLegalMindPipeline
print(f"Modules imported in {time.time() - t0:.2f}s", flush=True)

test_suite = TestLegalMindPipeline()
methods = [m for m in dir(test_suite) if m.startswith("test_")]

print(f"Running {len(methods)} unit tests...", flush=True)
passed = 0
failed = 0

for m in methods:
    func = getattr(test_suite, m)
    t_start = time.time()
    try:
        func()
        print(f"  [PASS] {m:<35} ({time.time() - t_start:.4f}s)", flush=True)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {m:<35} ERROR: {e}", flush=True)
        failed += 1

print(f"\nResults: {passed} passed, {failed} failed out of {len(methods)} tests.", flush=True)
if failed > 0:
    sys.exit(1)
sys.exit(0)
