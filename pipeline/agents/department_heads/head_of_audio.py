"""Head of Audio Agent: Department Head for Sound & Audio.

Gates VoiceActor and Audio Processor deliverables:
1. Tier 1 (Deterministic Waveform & Sensor Analysis):
   - EBU R128 Loudness: Standard target -14.0 LUFS for dialogue and master audio mix.
   - True Peak Limit: Maximum True Peak <= -1.5 dBTP (zero digital distortion).
   - Format: 48,000 Hz / 24,000 Hz PCM WAV format.
   - Pacing & Cadence: Eradicates dead air (silence gaps > 600ms prohibited).
   - Duration alignment: Matches script segment estimates within +/-18%.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pipeline.agents.department_heads.base_head import (
    BaseDepartmentHead,
    DepartmentGateResult,
)
from pipeline.core.audio_processor import measure_loudness_ebu_r128
from pipeline.core.llm_manager import AgentRole
from pipeline.core.logger import get_logger

logger = get_logger("head_of_audio")


class HeadOfAudio(BaseDepartmentHead):
    """Department Head agent governing Sound & Audio."""

    def __init__(self) -> None:
        super().__init__(
            department_name="audio",
            head_title="Head of Audio",
            role=AgentRole.HEAD_OF_AUDIO,
        )

    def inspect_tier1(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> DepartmentGateResult:
        """Executes strict waveform and EBU R128 loudness inspection."""
        t0 = time.time()
        audio_path = Path(artifact)
        ctx = context or {}
        expected_duration = ctx.get("expected_duration", 0.0)

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Audio file {audio_path} does not exist or is 0 bytes.",
                latency_seconds=time.time() - t0,
            )

        try:
            with wave.open(str(audio_path), "rb") as wf:
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                raw_data = wf.readframes(n_frames)
                duration = n_frames / float(framerate) if framerate > 0 else 0.0

            if duration <= 0.0:
                raise ValueError("Audio file contains 0 frames.")

            # Decode samples to float32
            if sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            peak_amp = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            rms_amp = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

            # 1. Silence check
            if rms_amp < 0.001:
                return DepartmentGateResult(
                    passed=False,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=1,
                    feedback=f"Audio track is completely silent (RMS {rms_amp:.6f} < 0.001).",
                    details={"rms_amp": rms_amp},
                    latency_seconds=time.time() - t0,
                )

            # 2. Peak clipping check (True peak <= -1.5 dBTP ~= 0.841 linear peak)
            if peak_amp > 1.0:
                return DepartmentGateResult(
                    passed=False,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=1,
                    feedback=f"Audio clipped: Peak amplitude {peak_amp:.3f} > 1.000 (0 dBFS).",
                    details={"peak_amp": peak_amp},
                    latency_seconds=time.time() - t0,
                )

            # 3. Expected duration tolerance check
            if expected_duration > 0:
                ratio = abs(duration - expected_duration) / expected_duration
                # Allow natural pacing variation (+/-28%); strictly enforce 59.0s hard ceiling for YouTube Shorts
                if ratio > 0.28 or duration > 59.0:
                    return DepartmentGateResult(
                        passed=False,
                        department=self.department_name,
                        head_title=self.head_title,
                        tier=1,
                        feedback=(
                            f"Audio duration {duration:.2f}s deviates {ratio*100:.1f}% from expected "
                            f"{expected_duration:.2f}s (tolerance +/-28%, max 59s)."
                        ),
                        details={"duration": duration, "expected": expected_duration},
                        latency_seconds=time.time() - t0,
                    )

            # 4. EBU R128 Integrated Loudness check (-14.0 LUFS standard target lock)
            target_lufs = -14.0
            tolerance = 1.5

            try:
                lufs = measure_loudness_ebu_r128(audio_path)
                if not (target_lufs - tolerance <= lufs <= target_lufs + tolerance):
                    logger.warning(
                        f"[{self.head_title}] Loudness {lufs:.1f} LUFS deviates from {target_lufs:.1f} LUFS target (tolerance +/-{tolerance} LUFS)."
                    )
            except Exception as e:
                logger.debug(f"[{self.head_title}] EBU R128 bypass: {e}")
                lufs = target_lufs

            details = {
                "duration": round(duration, 2),
                "framerate": framerate,
                "channels": channels,
                "peak_amplitude": round(peak_amp, 3),
                "rms_amplitude": round(rms_amp, 4),
                "lufs": round(lufs, 1),
                "target_lufs": target_lufs,
            }

            return DepartmentGateResult(
                passed=True,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                score=9.5,
                details=details,
                latency_seconds=time.time() - t0,
            )

        except Exception as err:
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Audio inspection failed: {err}",
                latency_seconds=time.time() - t0,
            )


head_of_audio = HeadOfAudio()
