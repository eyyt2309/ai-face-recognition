"""Webcam entry point; the logic lives in src.attention.attention_tracker."""

import sys
from pathlib import Path

# Allow `python src\main.py` as well as `python -m src.main`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attention.attention_tracker import runWebcam

if __name__ == "__main__":
    runWebcam()
