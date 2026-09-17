import json
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.dashboard.app import app
from pipeline.dashboard.database import (
    init_db,
    is_valid_youtube_url,
    normalize_youtube_url,
    set_job_youtube_url,
    set_video_youtube_url
)

@pytest.fixture
def client():
    app.config["TESTING"] = True
    init_db()
    with app.test_client() as client:
        yield client


def test_youtube_url_validator_and_normalizer():
    # Valid real YouTube URLs
    assert is_valid_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ") is True
    assert is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True

    # Invalid, mock, or 404-prone fake URLs
    assert is_valid_youtube_url("https://youtube.com/shorts/mock_123") is False
    assert is_valid_youtube_url("https://youtube.com/shorts/mock_job_99") is False
    assert is_valid_youtube_url("https://youtube.com/shorts/DRY_RUN_123") is False
    assert is_valid_youtube_url("https://youtube.com/shorts/short") is False
    assert is_valid_youtube_url(None) is False
    assert is_valid_youtube_url("") is False

    # Canonical normalization
    norm = normalize_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert norm == "https://youtube.com/shorts/dQw4w9WgXcQ"


def test_no_mock_youtube_urls_in_queue(client):
    res = client.get("/api/queue")
    assert res.status_code == 200
    data = res.get_json()
    assert "jobs" in data
    jobs = data["jobs"]
    for j in jobs:
        yt = j.get("youtube_url")
        if yt:
            assert "mock_" not in yt
            assert is_valid_youtube_url(yt) is True


def test_reject_invalid_or_mock_youtube_url_endpoint(client):
    res_q = client.get("/api/queue")
    jobs = res_q.get_json().get("jobs", [])
    jid = jobs[0]["id"] if jobs else "1"

    # Reject mock URL
    res = client.post(f"/api/jobs/{jid}/youtube_url", json={"youtube_url": "https://youtube.com/shorts/mock_404"})
    assert res.status_code == 400
    assert "Invalid" in res.get_json().get("error", "")

    # Reject malformed string
    res2 = client.post(f"/api/jobs/{jid}/youtube_url", json={"youtube_url": "not-a-url"})
    assert res2.status_code == 400


def test_set_and_retrieve_genuine_youtube_url(client):
    res_q = client.get("/api/queue")
    jobs = res_q.get_json().get("jobs", [])
    if jobs:
        jid = jobs[0]["id"]
        genuine_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        expected_norm = "https://youtube.com/shorts/dQw4w9WgXcQ"

        res = client.post(f"/api/jobs/{jid}/youtube_url", json={"youtube_url": genuine_url})
        assert res.status_code == 200
        assert res.get_json().get("youtube_url") == expected_norm

        # Verify reflected in queue
        res_check = client.get("/api/queue")
        updated_jobs = res_check.get_json().get("jobs", [])
        matched = next((j for j in updated_jobs if str(j["id"]) == str(jid)), None)
        assert matched is not None
        assert matched["youtube_url"] == expected_norm


def test_job_simulation_download_attachment(client):
    res_q = client.get("/api/queue")
    jobs = res_q.get_json().get("jobs", [])
    if jobs:
        jid = jobs[0]["id"]
        res = client.get(f"/api/jobs/{jid}/simulation/download")
        assert res.status_code == 200
        assert "attachment;" in res.headers.get("Content-Disposition", "")
        payload = json.loads(res.data.decode("utf-8"))
        assert "export_title" in payload
        assert "agents_executed" in payload
        assert len(payload["agents_executed"]) > 0
        assert "summary" in payload
        assert payload["summary"]["total_agents"] == len(payload["agents_executed"])


def test_assets_have_verified_urls_and_download(client):
    res = client.get("/api/assets")
    assert res.status_code == 200
    data = res.get_json()
    assert "final_artifacts" in data
    final_videos = [a for a in data["final_artifacts"] if a.get("type") == "video"]
    for v in final_videos:
        assert "simulation_download_url" in v
        yt = v.get("youtube_url")
        if yt:
            assert "mock_" not in yt
            assert is_valid_youtube_url(yt) is True


def test_video_youtube_link_endpoint(client):
    res = client.get("/api/assets")
    data = res.get_json()
    final_videos = [a for a in data.get("final_artifacts", []) if a.get("type") == "video"]
    if final_videos:
        vname = final_videos[0]["name"]
        genuine_url = "https://youtu.be/dQw4w9WgXcQ"
        expected_norm = "https://youtube.com/shorts/dQw4w9WgXcQ"

        res_post = client.post(f"/api/videos/{vname}/youtube_url", json={"youtube_url": genuine_url})
        assert res_post.status_code == 200
        assert res_post.get_json().get("youtube_url") == expected_norm

        res_sim = client.get(f"/api/videos/{vname}/simulation")
        assert res_sim.status_code == 200
        sim_data = res_sim.get_json()
        assert sim_data.get("youtube_url") == expected_norm

        res_dl = client.get(f"/api/videos/{vname}/simulation/download")
        assert res_dl.status_code == 200
        assert "attachment;" in res_dl.headers.get("Content-Disposition", "")


def test_ceo_dashboard_endpoints():
    from ceo_dashboard import app as ceo_app
    ceo_app.config["TESTING"] = True
    with ceo_app.test_client() as c:
        res = c.get("/status")
        assert res.status_code == 200
        data = res.get_json()
        assert "status" in data
        assert "latest_run" in data

        res_assets = c.get("/api/assets")
        assert res_assets.status_code == 200
        assets = res_assets.get_json()
        final_videos = [a for a in assets.get("final_artifacts", []) if a.get("type") == "video"]
        if final_videos:
            vname = final_videos[0]["name"]
            res_vdl = c.get(f"/api/videos/{vname}/simulation/download")
            assert res_vdl.status_code == 200
            assert "attachment;" in res_vdl.headers.get("Content-Disposition", "")
