"""Live Simulation Verification Script for Mission G.

Submits 3 jobs rapidly to /api/start and monitors sequential execution.
"""

import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:5000"

def post_job(topic, niche="tech"):
    req = urllib.request.Request(
        f"{BASE_URL}/api/start",
        data=json.dumps({"topic": topic, "niche": niche}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_queue():
    with urllib.request.urlopen(f"{BASE_URL}/api/queue") as resp:
        return json.loads(resp.read().decode("utf-8")).get("jobs", [])

def get_logs():
    with urllib.request.urlopen(f"{BASE_URL}/api/logs?lines=5") as resp:
        return json.loads(resp.read().decode("utf-8")).get("logs", [])

def main():
    print("=" * 70)
    print("LIVE SIMULATION: 3 RAPID JOB SUBMISSIONS TO COMMAND CENTER")
    print("=" * 70)

    # 1. Submit 3 jobs in rapid succession
    jobs_to_submit = [
        "Dark Matter & Galactic Rotation",
        "Neural Interfaces & Memory Storage",
        "CRISPR Gene Editing for Immortality"
    ]
    submitted = []
    for topic in jobs_to_submit:
        res = post_job(topic)
        submitted.append(res["job_id"])
        print(f"[SUBMIT] Enqueued job #{res['job_id']}: '{topic}' -> Status: {res['status']}")

    print("\n[VERIFY] Monitoring sequential execution across queue (sampling every 2s)...")
    completed_ids = set()
    max_duration = 50.0
    start_t = time.time()

    running_history = []

    while time.time() - start_t < max_duration:
        all_jobs = get_queue()
        job_map = {j["id"]: j for j in all_jobs}

        active_running = [j for j in all_jobs if j["status"] == "RUNNING"]
        running_count = len(active_running)
        running_history.append(running_count)

        # Invariant: NEVER more than 1 job in RUNNING state (Rule 1 & Rule 7)
        if running_count > 1:
            raise RuntimeError(f"RULE VIOLATION: {running_count} jobs running simultaneously: {[j['id'] for j in active_running]}")

        target_statuses = [f"#{jid}: {job_map[jid]['status']}" for jid in submitted if jid in job_map]
        elapsed = round(time.time() - start_t, 1)

        print(f"[T+{elapsed:>4}s] Queue Status: {' | '.join(target_statuses)} (Active Running: {running_count})")

        # Check if all 3 submitted jobs completed
        if all(job_map.get(jid, {}).get("status") in ("COMPLETED", "SUCCESS") for jid in submitted):
            print("\n" + "=" * 70)
            print("SUCCESS: All 3 jobs executed strictly sequentially to completion!")
            print(f"Max Concurrent Running Jobs: {max(running_history)} (Strictly <= 1)")
            print("=" * 70)
            logs = get_logs()
            print("\nRecent Logs:")
            print("\n".join(logs[-4:]))
            return 0

        time.sleep(2.0)

    raise TimeoutError("Jobs did not complete within max duration.")

if __name__ == "__main__":
    main()
