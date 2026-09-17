"""Test pipeline.lock live pause and resume."""

import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:5000"
LOCK = "pipeline.lock"

# 1. Create pipeline.lock with alive PID
with open(LOCK, "w") as f:
    json.dump({"pid": os.getpid(), "started_at": "2026-09-17T00:00:00Z"}, f)
print("[TEST] Acquired simulated manual CLI lock:", LOCK)

try:
    # 2. Enqueue job
    req = urllib.request.Request(
        f"{BASE}/api/start",
        data=json.dumps({"topic": "Lock Pause Verification Job"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    jid = res["job_id"]
    print(f"[TEST] Enqueued job #{jid}")

    # 3. Verify worker pauses for 6 seconds while lock is present
    time.sleep(6.0)
    with urllib.request.urlopen(f"{BASE}/api/queue") as resp:
        jobs = json.loads(resp.read().decode("utf-8"))["jobs"]
    j = next(x for x in jobs if x["id"] == jid)
    print(f"[TEST] Status while locked (QUEUED expected): {j['status']}")
    assert j["status"] == "QUEUED", "Job ran despite pipeline.lock being active!"

    # 4. Release lock
    if os.path.exists(LOCK):
        os.remove(LOCK)
    print("[TEST] Released manual CLI lock.")

    # 5. Wait for worker to pick up and complete job
    time.sleep(15.0)
    with urllib.request.urlopen(f"{BASE}/api/queue") as resp:
        jobs = json.loads(resp.read().decode("utf-8"))["jobs"]
    j = next(x for x in jobs if x["id"] == jid)
    print(f"[TEST] Status after lock release (COMPLETED expected): {j['status']}")
    assert j["status"] in ("RUNNING", "COMPLETED"), f"Job did not resume: {j['status']}"
    print("[TEST] SUCCESS: RULE 7 VERIFICATION PASSED!")

finally:
    if os.path.exists(LOCK):
        os.remove(LOCK)
