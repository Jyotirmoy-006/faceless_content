import json, subprocess, tempfile, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.audio_processor import detect_audio_silences

fixed_video = Path("pipeline/output/20260916_180043_Why_Your_Smart_Bulb_Might_Be_L_fixed.mp4")
orig_video = Path("pipeline/output/20260916_180043_Why_Your_Smart_Bulb_Might_Be_L_original_unfixed.mp4")

# 1. FFprobe metadata on fixed video
cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,duration,width,height", "-of", "json", str(fixed_video)]
res = subprocess.run(cmd, capture_output=True, text=True, check=True)
print("=== 1. FFPROBE METADATA (FIXED VIDEO) ===")
print(res.stdout)

# 2. Compare video durations
cmd_orig = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(orig_video)]
res_orig = subprocess.run(cmd_orig, capture_output=True, text=True, check=True)
dur_orig = float(json.loads(res_orig.stdout)["format"]["duration"])
dur_fixed = float(json.loads(res.stdout)["format"]["duration"])
print(f"Original Video Duration: {dur_orig:.3f} s")
print(f"Fixed Video Duration:    {dur_fixed:.3f} s")
print(f"Net Duration Saved:      {dur_orig - dur_fixed:.3f} s ({(dur_orig - dur_fixed)/dur_orig*100:.1f}% compression)")

# 3. Silence Detection (>150ms at -40dB) on Fixed Video Audio
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
    tmp_wav = tf.name
subprocess.run(["ffmpeg", "-y", "-i", str(fixed_video), "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", tmp_wav], capture_output=True, check=True)

silences_fixed = detect_audio_silences(tmp_wav, min_silence_len=150, silence_thresh=-40.0)
print(f"\n=== 3. SILENCE DETECTION ON FIXED VIDEO (>150ms at -40dB) ===")
print(f"Silences > 150ms: {len(silences_fixed)}")
print(f"Detected Intervals: {silences_fixed}")

# 4. Check all silences > 80ms
silences_80ms = detect_audio_silences(tmp_wav, min_silence_len=80, silence_thresh=-40.0)
print(f"\nAudit of pauses >80ms (intended sentence boundaries):")
for s, e in silences_80ms:
    print(f"  Pause from {s:5d}ms to {e:5d}ms -> {e - s:3d}ms")
