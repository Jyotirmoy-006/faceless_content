"""Isolated worker subprocess scripts.

Per Rule 1 (GPU MEMORY): Every GPU-bound stage (Whisper, ComfyUI, NVENC render)
MUST run as an isolated subprocess that exits when the stage completes.
"""
