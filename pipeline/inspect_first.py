import subprocess, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.audio_processor import detect_audio_silences

video_path = Path("pipeline/output/20260916_180043_Why_Your_Smart_Bulb_Might_Be_L.mp4")
tmp_wav = Path("pipeline/assets_cache/first_video_audio.wav")

cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(tmp_wav)]
subprocess.run(cmd, capture_output=True, check=True)

# 1. Silences in original first video
silences = detect_audio_silences(tmp_wav, min_silence_len=200, silence_thresh=-40.0)
print(f"Total >200ms silences in original video: {len(silences)}")
for s, e in silences:
    print(f"  Silence from {s}ms to {e}ms -> duration: {e - s}ms")

# 2. Whisper transcription
from pipeline.workers.whisper_worker import transcribe_audio
res = transcribe_audio(tmp_wav, device="cuda")
print("\n=== TRANSCRIBED TEXT OF FIRST VIDEO ===")
for seg in res.get("segments", []):
    print(f"  [{seg['start']:.2f}s -> {seg['end']:.2f}s]: {seg['text']}")
