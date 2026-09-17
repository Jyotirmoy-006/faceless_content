"""Re-renders the first generated video with all pipeline fixes applied:
- Canonical 1080x1920 30fps CFR compose-mode normalization (Rule 9)
- Word-level styled hard-burned captions (Rule 10)
- High-retention zero-breath audio pacing (+15% rate, internal silence eradication, Rule 11)
- Closed-GOP NVENC encoding with Faststart moov atom
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.schema import Script, ScriptSegment
from pipeline.agents.director import orchestrate_video

def build_first_video_script() -> Script:
    """Builds the structured Script object matching the exact content of the first video."""
    return Script(
        topic="The Artificial Intelligence Breakthrough",
        niche="tech",
        target_duration=30,
        hook="Artificial intelligence just crossed a threshold nobody thought was possible this year.",
        segments=[
            ScriptSegment(
                segment_index=1,
                narration="Artificial intelligence just crossed a threshold nobody thought was possible this year.",
                visual_query="glowing digital neural network silicon microchip data stream futuristic 4k",
                duration_seconds=5.0
            ),
            ScriptSegment(
                segment_index=2,
                narration="Engineers discovered that scaling neural models unlocked capabilities not even in the training data.",
                visual_query="cyberpunk holographic coding screen data visualization dark aesthetic",
                duration_seconds=6.0
            ),
            ScriptSegment(
                segment_index=3,
                narration="Automated reasoning systems are now solving problems in minutes that once took human teams months.",
                visual_query="cinematic close up of glowing futuristic interface neon holographic",
                duration_seconds=6.0
            ),
            ScriptSegment(
                segment_index=4,
                narration="The speed of adoption means the tools you rely on tomorrow are being written right now.",
                visual_query="futuristic city skyline at night drone flying cybernetic glow",
                duration_seconds=5.0
            ),
            ScriptSegment(
                segment_index=5,
                narration="Subscribe to stay ahead of the next tech wave.",
                visual_query="cinematic macro shot of glowing cybernetic technology",
                duration_seconds=3.0
            ),
        ],
        cta="Subscribe to stay ahead of the next tech wave."
    )

def main():
    script = build_first_video_script()
    output_dir = ROOT_DIR / "pipeline" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_output_path = output_dir / "20260916_180043_Why_Your_Smart_Bulb_Might_Be_L_fixed.mp4"
    canonical_output_path = output_dir / "20260916_180043_Why_Your_Smart_Bulb_Might_Be_L.mp4"
    
    print("=" * 80)
    print("STARTING FULL RE-RENDER OF FIRST VIDEO (WITH MISSIONS A-F FIXES)")
    print(f"Target Output: {fixed_output_path}")
    print("=" * 80)
    
    t0 = time.time()
    rendered_path = orchestrate_video(
        script=script,
        output_path=fixed_output_path,
        voice="en-US-ChristopherNeural"
    )
    render_time = time.time() - t0
    
    print("\n" + "=" * 80)
    print(f"RE-RENDER COMPLETE in {render_time:.2f}s!")
    print(f"Fixed Video Path: {rendered_path} ({rendered_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print("=" * 80)
    
    # Also update canonical path so upload_video.py points to the fixed video
    import shutil
    shutil.copy2(str(fixed_output_path), str(canonical_output_path))
    print(f"Updated canonical video: {canonical_output_path}")

if __name__ == "__main__":
    main()
