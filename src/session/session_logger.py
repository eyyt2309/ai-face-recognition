"""Writes a per-frame CSV log of attention and gaze for later analysis."""

import csv
import json
from datetime import datetime
from pathlib import Path

from ..config import PROJECT_ROOT

SESSIONS_DIR = PROJECT_ROOT / "sessions"

COLUMNS = [
    "t",              # seconds since the session started
    "attention",      # 0 attentive, 1 inattentive, empty if unknown
    "faces",
    "hands",
    "phone",          # 1 if a phone was detected
    "pose",           # forward / left / right / down, empty if no face
    "gaze_x",         # normalised screen position, empty if unavailable
    "gaze_y",
    "off_screen",     # 1 if the gaze estimate lies outside the screen
    "marker"          # section label added by the user, usually empty
]


def newSessionPath(directory=None):
    directory = Path(directory or SESSIONS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"session_{stamp}.csv"


def _blank(value, digits=4):
    return "" if value is None else round(float(value), digits)


class SessionLogger:
    """Append-only CSV writer; one row per logged frame."""

    def __init__(self, path=None, metadata=None):
        self.path = Path(path) if path else newSessionPath()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = 0

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

        self.metadata = {"started": datetime.now().isoformat(timespec="seconds")}
        self.metadata.update(metadata or {})
        self.metaPath = self.path.with_suffix(".meta.json")
        self._writeMetadata()

    def log(self, t, attention=None, faces=0, hands=0, phone=False,
            pose=None, gaze=None, offScreen=False, marker=""):
        gazeX, gazeY = (None, None) if gaze is None else (gaze[0], gaze[1])

        self._writer.writerow([
            round(float(t), 3),
            "" if attention is None else int(attention),
            int(faces),
            int(hands),
            int(bool(phone)),
            pose or "",
            _blank(gazeX),
            _blank(gazeY),
            int(bool(offScreen)),
            marker
        ])
        self.rows += 1

    def finish(self, **extra):
        """Flush the log and store closing metadata (duration, fps, ...)."""

        self.metadata.update(extra)
        self.metadata["rows"] = self.rows
        self._writeMetadata()
        self._file.flush()

    def _writeMetadata(self):
        self.metaPath.write_text(
            json.dumps(self.metadata, indent=2),
            encoding="utf-8"
        )

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, excType, excValue, traceback):
        self.close()
