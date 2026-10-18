"""Calibration mode: record reference points on the jaws by placing the marker and pressing Enter.

Flow (in `drag` mode):
    Enter            start calibration; the screen asks for the first reference point
    drag marker      put the green marker on that point of the jaw
    Enter            record it, the next point is requested
    after the last   all points are written to CALIBRATION_FILE (world + both jaw frames, mm)

What the points are used for is up to you: e.g. compare them with the same points measured on
the real phantom to find where the real jaw sits relative to the robot.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .jaw import LOWER, UPPER

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.json"

REFERENCE_POINTS = (
    "lower incisor edge, middle",
    "lower last molar, patient left",
    "lower last molar, patient right",
    "upper incisor edge, middle",
)


class CalibrationRecorder:
    """Walks through REFERENCE_POINTS, one Enter press per point."""

    def __init__(self):
        self.active = False
        self.recorded: dict[str, np.ndarray] = {}
        self.last_saved: Path | None = None

    def press_enter(self, marker_world: np.ndarray) -> None:
        """Start calibration, or record the current marker position for the requested point."""
        if not self.active:
            self.active = True
            self.recorded = {}
            return
        self.recorded[self.current_point()] = marker_world.copy()
        if len(self.recorded) == len(REFERENCE_POINTS):
            self.last_saved = self._save()
            self.active = False

    def current_point(self) -> str:
        return REFERENCE_POINTS[len(self.recorded)]

    def status(self) -> str:
        """One line for the on-screen text."""
        if self.active:
            step = len(self.recorded) + 1
            return f"CALIBRATION {step}/{len(REFERENCE_POINTS)}: put marker on {self.current_point()}, press Enter"
        if self.last_saved:
            return f"calibration saved to {self.last_saved.name}; Enter = start again"
        return "Enter = start calibration"

    def _save(self) -> Path:
        points = {
            name: {
                "world_mm": _mm(point * 1000),
                "lower_stl_mm": _mm(LOWER.to_stl(point)),
                "upper_stl_mm": _mm(UPPER.to_stl(point)),
            }
            for name, point in self.recorded.items()
        }
        CALIBRATION_FILE.write_text(json.dumps({"recorded_at": datetime.now().isoformat(timespec="seconds"), "points": points}, indent=2))
        return CALIBRATION_FILE


def _mm(vector: np.ndarray) -> list[float]:
    return [round(float(v), 2) for v in vector]
