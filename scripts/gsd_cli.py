#!/usr/bin/env python3
"""GSD (Get Stuff Done) Command Line Interface.

Provides terminal initialization, status tracking, and roadmap management
compatible with Antigravity AI pair programming.
"""

import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def print_banner():
    print("\n\033[96m" + "=" * 62)
    print("             GSD (Get Stuff Done) CLI Engine              ")
    print("=" * 62 + "\033[0m\n")

def init_planning(workspace_dir: Path):
    print_banner()
    planning_dir = workspace_dir / ".planning"
    planning_dir.mkdir(parents=True, exist_ok=True)
    print(f" \033[92m[+]\033[0m Planning directory verified: .planning/")

    # 1. config.json
    config_file = planning_dir / "config.json"
    if not config_file.exists():
        config_data = {
            "workflow": "gsd-sdd",
            "version": "2.0.0",
            "autonomous": False,
            "commit_per_plan": True,
            "verification_required": True
        }
        config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        print(" \033[92m[+]\033[0m Created .planning/config.json")

    # 2. STATE.md
    state_file = planning_dir / "STATE.md"
    if not state_file.exists():
        state_content = """# GSD Project State

- **Current Milestone:** v1.0.0 — Production-Grade Faceless Automation Engine
- **Active Phase:** Phase 1: High-Retention Visual & Audio Foundation (Complete)
- **Next Phase:** Phase 2: Autonomous Production & Scaling
- **Status:** READY
"""
        state_file.write_text(state_content, encoding="utf-8")
        print(" \033[92m[+]\033[0m Created .planning/STATE.md")

    # 3. PROJECT.md
    project_file = planning_dir / "PROJECT.md"
    if not project_file.exists():
        project_content = """# Project Specification: Faceless Automation Engine

Autonomous AI-driven studio that generates, animates, narrates, captions, and renders short-form videos with zero human latency.

## Architecture
- **Voice Actor:** Edge-TTS with dynamic SSML prosody (+8% rate, -2Hz pitch on hooks) and EBU R128 (-14.0 LUFS) normalization.
- **Art Director:** Local ComfyUI SD1.5/SDXL-Turbo keyframe generation under GPU lock, anti-drift semantic relevance gating (>= 0.78).
- **Editor:** Beat-synced micro-cuts (1.2s - 2.2s, max 2.5s) with alternating zoompan camera momentum and NVENC encoding.
- **Subtitles:** Kinetic, animated ASS captions at Y=1400 with 110% word scale pop in Vivid Yellow (&H0000FFFF&).
"""
        project_file.write_text(project_content, encoding="utf-8")
        print(" \033[92m[+]\033[0m Created .planning/PROJECT.md")

    # 4. ROADMAP.md
    roadmap_file = planning_dir / "ROADMAP.md"
    if not roadmap_file.exists():
        roadmap_content = """# Project Roadmap

## Phase 1: High-Retention Visual & Audio Foundation
- [x] Eliminate stock footage drift with ComfyUI primary generation.
- [x] Beat-synced micro-cuts bounded between 1.2s and 2.2s (max 2.5s).
- [x] Dynamic SSML narration cadence with punctuation breaks.
- [x] Kinetic ASS captions centered at Y=1400 with 110% word-level highlight.

## Phase 2: Autonomous Production & Scaling
- [ ] End-to-end multi-job overnight scheduler.
- [ ] Multi-platform distribution and analytics telemetry.
"""
        roadmap_file.write_text(roadmap_content, encoding="utf-8")
        print(" \033[92m[+]\033[0m Created .planning/ROADMAP.md")

    print("\n \033[92m[SUCCESS] GSD Framework successfully initialized!\033[0m")
    print("\n \033[96mNext steps in Antigravity Chat:\033[0m")
    print("   - Type \033[97m/gsd-plan-phase 2\033[0m   (to plan the next milestone phase)")
    print("   - Type \033[97m/gsd-progress\033[0m       (to check roadmap execution)")
    print("   - Type \033[97m/gsd-help\033[0m           (to list all 30+ GSD commands)\n")

def show_status(workspace_dir: Path):
    print_banner()
    state_file = workspace_dir / ".planning" / "STATE.md"
    if state_file.exists():
        print(state_file.read_text(encoding="utf-8"))
    else:
        print(" \033[93m[!] No GSD state found. Run 'gsd init' first to initialize.\033[0m")

def show_progress(workspace_dir: Path):
    print_banner()
    roadmap_file = workspace_dir / ".planning" / "ROADMAP.md"
    if roadmap_file.exists():
        print(roadmap_file.read_text(encoding="utf-8"))
    else:
        print(" \033[93m[!] No GSD roadmap found. Run 'gsd init' first.\033[0m")

def show_help():
    print_banner()
    print("\033[96mCLI Usage:\033[0m")
    print("  gsd init         - Initialize .planning directory and roadmap")
    print("  gsd status       - Show current milestone and phase state")
    print("  gsd progress     - Display roadmap completion progress")
    print("  gsd help         - Show this help message\n")
    print("\033[96mAvailable AI Slash Commands in Antigravity:\033[0m")
    print("  /gsd-new-project     - Initialize brand new project")
    print("  /gsd-plan-phase [N]  - Plan phase tasks")
    print("  /gsd-execute-phase   - Run phase plans")
    print("  /gsd-progress        - Unified status check")
    print("  /gsd-verify-work     - Run conversational acceptance tests")
    print("  /gsd-help            - Full GSD cheatsheet\n")

def main():
    workspace_dir = Path.cwd()
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"
    
    if cmd == "init":
        init_planning(workspace_dir)
    elif cmd == "status":
        show_status(workspace_dir)
    elif cmd == "progress":
        show_progress(workspace_dir)
    else:
        show_help()

if __name__ == "__main__":
    main()
