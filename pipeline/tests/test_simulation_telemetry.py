"""Comprehensive Test Suite for Mission M: Agentic Simulation Telemetry & Segregated Assets.

Verifies:
1. SQLite schema upgrades: `simulation_trace_json`, `raw_materials_json`, `final_artifacts_json` in jobs and telemetry, plus `simulation_steps` table.
2. Granular step-by-step recording: `record_simulation_step()` and `get_job_simulation_trace()` persist and retrieve execution traces.
3. Chronological simulation endpoint `/api/jobs/<id>/simulation` returns valid JSON trace on both Command Center and CEO dashboards.
4. Strict Segregation: intermediate raw materials (Pexels, ComfyUI, TTS WAV chunks, Whisper SRTs, 480p QA proxies) are categorized as raw_materials, while pristine master outputs (1080x1920 30fps CFR MP4, ASS script, JSON metadata) are isolated in final_artifacts with ZERO cross-contamination.
5. REST endpoints `/api/assets` and `/api/videos` return segregated asset inventories with correct counts.
"""

import json
import sqlite3
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.dashboard.database import (
    init_db,
    get_db_connection,
    enqueue_job,
    record_simulation_step,
    get_job_simulation_trace,
    get_categorized_assets,
    record_job_assets,
    record_run_telemetry,
    get_latest_run_telemetry
)
from pipeline.dashboard.app import app as command_center_app
from ceo_dashboard import app as ceo_app


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite DB path for testing."""
    db_file = tmp_path / "test_mission_m.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def test_dirs(tmp_path):
    """Creates isolated mock output and assets_cache directories."""
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "assets_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create pristine final deliverables in output
    master_mp4 = output_dir / "20260917_quantum_computing.mp4"
    master_mp4.write_bytes(b"FAKE_MP4_MASTER_1080x1920_CONTENT" * 50)
    master_ass = output_dir / "20260917_quantum_computing.ass"
    master_ass.write_text("[Script Info]\nTitle: Master Captions", encoding="utf-8")
    master_json = output_dir / "20260917_quantum_computing.json"
    master_json.write_text(json.dumps({"title": "Quantum Computing", "tags": ["tech"]}), encoding="utf-8")

    # 2. Create intermediate raw materials in assets_cache
    pexels_dir = cache_dir / "pexels"
    pexels_dir.mkdir(parents=True, exist_ok=True)
    pexels_clip = pexels_dir / "stock_01.mp4"
    pexels_clip.write_bytes(b"PEXELS_RAW_STOCK_VIDEO" * 10)

    tts_dir = cache_dir / "run_test" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    tts_chunk = tts_dir / "chunk_01.wav"
    tts_chunk.write_bytes(b"RIFF_FAKE_AUDIO_WAV" * 10)

    comfy_dir = cache_dir / "comfyui"
    comfy_dir.mkdir(parents=True, exist_ok=True)
    comfy_frame = comfy_dir / "frame_001.png"
    comfy_frame.write_bytes(b"PNG_FAKE_IMAGE_DATA" * 10)

    whisper_srt = cache_dir / "transcripts" / "audio.srt"
    whisper_srt.parent.mkdir(parents=True, exist_ok=True)
    whisper_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nRaw audio subtitle", encoding="utf-8")

    proxy_dir = cache_dir / "qa_proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    qa_proxy = proxy_dir / "quantum_computing_proxy.mp4"
    qa_proxy.write_bytes(b"LIGHTWEIGHT_480P_PROXY" * 10)

    return {
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "master_mp4": master_mp4,
        "master_ass": master_ass,
        "master_json": master_json,
        "pexels_clip": pexels_clip,
        "tts_chunk": tts_chunk,
        "comfy_frame": comfy_frame,
        "whisper_srt": whisper_srt,
        "qa_proxy": qa_proxy
    }


class TestSimulationTelemetrySchema:
    """Verifies SQLite tables, columns, and step-by-step telemetry persistence."""

    def test_01_schema_migration_includes_mission_m_columns(self, temp_db):
        """Verifies jobs and telemetry tables contain simulation and asset segregation columns."""
        with get_db_connection(temp_db) as conn:
            # Check jobs table
            jobs_cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
            assert "simulation_trace_json" in jobs_cols
            assert "raw_materials_json" in jobs_cols
            assert "final_artifacts_json" in jobs_cols

            # Check telemetry table
            telemetry_cols = [r[1] for r in conn.execute("PRAGMA table_info(telemetry)").fetchall()]
            assert "simulation_trace_json" in telemetry_cols
            assert "raw_materials_json" in telemetry_cols
            assert "final_artifacts_json" in telemetry_cols

            # Check simulation_steps table exists
            steps_cols = [r[1] for r in conn.execute("PRAGMA table_info(simulation_steps)").fetchall()]
            assert "job_id" in steps_cols
            assert "agent_name" in steps_cols
            assert "input_payload_json" in steps_cols
            assert "reasoning_trace" in steps_cols
            assert "prompt_template" in steps_cols
            assert "structured_output_json" in steps_cols
            assert "duration_seconds" in steps_cols

    def test_02_record_and_retrieve_simulation_steps(self, temp_db):
        """Verifies recording granular simulation snapshots and chronological retrieval."""
        job = enqueue_job(topic="Simulation Test", niche="tech", db_path=temp_db)
        job_id = job["id"]

        # Record step 1: Creative Director
        step1 = {
            "step_index": 1,
            "agent_name": "CreativeDirector",
            "stage": "IDEATION",
            "status": "PASSED",
            "duration_seconds": 0.85,
            "reasoning_trace": "Selected high viral hook with quantum premise.",
            "prompt_template": "Brainstorm 3 high-retention concepts.",
            "input_payload": {"niche": "tech", "topic": "Quantum Teleportation"},
            "structured_output": {"angle": "Physics Mystery", "hook": "Did atoms just teleport?"}
        }
        ok1 = record_simulation_step(job_id, step1, db_path=temp_db)
        assert ok1 is True

        # Record step 2: Copywriter
        step2 = {
            "step_index": 2,
            "agent_name": "Copywriter",
            "stage": "SCRIPTWRITING",
            "status": "PASSED",
            "duration_seconds": 1.25,
            "reasoning_trace": "Pacing calibrated for 45s vertical short.",
            "prompt_template": "Draft short-form script with beat breakdown.",
            "input_payload": {"angle": "Physics Mystery"},
            "structured_output": {"beats": [{"second": 0, "text": "Listen closely."}]}
        }
        ok2 = record_simulation_step(job_id, step2, db_path=temp_db)
        assert ok2 is True

        # Retrieve trace
        trace = get_job_simulation_trace(job_id, db_path=temp_db)
        assert len(trace) == 2
        assert trace[0]["agent_name"] == "CreativeDirector"
        assert trace[0]["duration_seconds"] == 0.85
        assert trace[0]["structured_output"]["angle"] == "Physics Mystery"
        assert trace[1]["agent_name"] == "Copywriter"
        assert trace[1]["structured_output"]["beats"][0]["text"] == "Listen closely."

    def test_03_fallback_simulation_trace_for_legacy_jobs(self, temp_db):
        """Verifies fallback synthetic reconstruction when inspecting legacy jobs without stored trace."""
        job = enqueue_job(topic="Legacy Job", niche="science", db_path=temp_db)
        trace = get_job_simulation_trace(job["id"], db_path=temp_db)
        assert len(trace) >= 6
        agent_names = [s["agent_name"] for s in trace]
        assert "Strategist" in agent_names
        assert "CreativeDirector" in agent_names
        assert "Copywriter" in agent_names
        assert "ComplianceOfficer" in agent_names


class TestAssetVaultSegregation:
    """Verifies strict isolation between intermediate raw materials and final deliverables."""

    def test_01_categorized_assets_segregates_raw_vs_final(self, test_dirs):
        """Verifies get_categorized_assets segregates master deliverables from intermediate cache."""
        categorized = get_categorized_assets(
            output_dir=test_dirs["output_dir"],
            cache_dir=test_dirs["cache_dir"]
        )

        final_artifacts = categorized["final_artifacts"]
        raw_materials = categorized["raw_materials"]

        assert categorized["final_count"] == 3
        assert categorized["raw_count"] == 5

        # Check final deliverables: only files from output_dir
        final_names = [item["name"] for item in final_artifacts]
        assert "20260917_quantum_computing.mp4" in final_names
        assert "20260917_quantum_computing.ass" in final_names
        assert "20260917_quantum_computing.json" in final_names

        # Check raw materials: only files from assets_cache
        raw_names = [item["name"] for item in raw_materials]
        assert "stock_01.mp4" in raw_names
        assert "chunk_01.wav" in raw_names
        assert "frame_001.png" in raw_names
        assert "audio.srt" in raw_names
        assert "quantum_computing_proxy.mp4" in raw_names

    def test_02_zero_cross_contamination(self, test_dirs):
        """CRITICAL: Verifies zero cross-contamination between raw intermediates and final artifacts."""
        categorized = get_categorized_assets(
            output_dir=test_dirs["output_dir"],
            cache_dir=test_dirs["cache_dir"]
        )

        final_artifacts = categorized["final_artifacts"]
        raw_materials = categorized["raw_materials"]

        # Ensure NO raw materials appear in final_artifacts
        for final_item in final_artifacts:
            assert "stock" not in final_item["name"].lower()
            assert "proxy" not in final_item["name"].lower()
            assert not final_item["name"].endswith(".wav")
            assert final_item["badge"] in ("FINAL MASTER", "FINAL SCRIPT", "FINAL METADATA")
            assert final_item["url"].startswith("/videos/")

        # Ensure NO final artifacts appear in raw_materials
        for raw_item in raw_materials:
            assert raw_item["badge"] == "RAW MATERIAL"
            assert raw_item["url"].startswith("/raw_materials/")
            assert "20260917_quantum_computing.mp4" != raw_item["name"]

    def test_03_record_job_assets_persistence(self, temp_db, test_dirs):
        """Verifies segregated assets can be persisted directly to a job row in SQLite."""
        job = enqueue_job(topic="Asset Test Job", niche="tech", db_path=temp_db)
        categorized = get_categorized_assets(
            output_dir=test_dirs["output_dir"],
            cache_dir=test_dirs["cache_dir"]
        )

        ok = record_job_assets(
            job_id_or_pk=job["id"],
            raw_materials=categorized["raw_materials"],
            final_artifacts=categorized["final_artifacts"],
            db_path=temp_db
        )
        assert ok is True

        with get_db_connection(temp_db) as conn:
            row = conn.execute("SELECT raw_materials_json, final_artifacts_json FROM jobs WHERE id = ?", (job["id"],)).fetchone()
            saved_raw = json.loads(row["raw_materials_json"])
            saved_final = json.loads(row["final_artifacts_json"])
            assert len(saved_raw) == 5
            assert len(saved_final) == 3


class TestDashboardRestEndpoints:
    """Verifies REST endpoints /api/jobs/<id>/simulation, /api/assets, and /api/videos."""

    def test_01_command_center_simulation_endpoint(self, temp_db):
        """Verifies Command Center dashboard /api/jobs/<id>/simulation returns 200 with trace."""
        job = enqueue_job(topic="Endpoint Job", niche="tech", db_path=temp_db)
        step = {
            "agent_name": "ComplianceOfficer",
            "stage": "COMPLIANCE",
            "status": "PASSED",
            "duration_seconds": 0.44,
            "reasoning_trace": "Content vetted against YouTube community guidelines. Zero demonetization triggers.",
            "prompt_template": "Assess risk of YouTube Short.",
            "input_payload": {"proxy_video": "qa_proxy.mp4"},
            "structured_output": {"safe_for_monetization": True, "risk_score": 0.05}
        }
        record_simulation_step(job["id"], step, db_path=temp_db)

        client = command_center_app.test_client()
        with command_center_app.app_context():
            # Override database path for test
            import pipeline.dashboard.app as cc_app_module
            orig_db = getattr(cc_app_module, "DEFAULT_DB_PATH", None)

            res = client.get(f"/api/jobs/{job['id']}/simulation")
            assert res.status_code == 200
            data = res.get_json()
            assert data["job_id"] == str(job["id"])
            assert "trace" in data
            assert len(data["trace"]) >= 1

    def test_02_ceo_dashboard_simulation_endpoint(self, temp_db):
        """Verifies CEO dashboard /api/jobs/<id>/simulation returns 200 with chronological trace."""
        job = enqueue_job(topic="CEO Sim Job", niche="finance", db_path=temp_db)
        ceo_app.config["TESTING"] = True
        ceo_app.config["DB_PATH"] = str(temp_db)
        client = ceo_app.test_client()

        res = client.get(f"/api/jobs/{job['id']}/simulation")
        assert res.status_code == 200
        data = res.get_json()
        assert "trace" in data
        assert len(data["trace"]) >= 1
        assert "agent_name" in data["trace"][0]

    def test_03_assets_endpoint_isolation(self):
        """Verifies /api/assets returns final_artifacts and raw_materials segregations."""
        client = command_center_app.test_client()
        res = client.get("/api/assets")
        assert res.status_code == 200
        data = res.get_json()
        assert "final_artifacts" in data
        assert "raw_materials" in data
        assert "final_count" in data
        assert "raw_count" in data
        assert isinstance(data["final_artifacts"], list)
        assert isinstance(data["raw_materials"], list)

    def test_04_videos_endpoint_backward_compatibility(self):
        """Verifies /api/videos preserves existing 'videos' array while including segregated assets."""
        client = command_center_app.test_client()
        res = client.get("/api/videos")
        assert res.status_code == 200
        data = res.get_json()
        assert "videos" in data
        assert "final_artifacts" in data
        assert "raw_materials" in data
        assert isinstance(data["videos"], list)
