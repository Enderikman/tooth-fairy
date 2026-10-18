"""Calibration: three points on the lower jaw define the mouth frame and the jaw size.

    1. incisal edge between 31 and 41 (on top of the tooth)   -> origin of the mouth frame
    2. tooth 36, lingual side, at the gum line                 -> patient left end of the arch (+x)
    3. tooth 46, lingual side, at the gum line                 -> patient right end of the arch (-x)

Flow in the viewer (`drag` mode):
    Enter            start calibration; the screen asks for the first point
    drag marker      put the green marker on that point
    Enter            record it, the next point is requested
    after the last   the points are written to calibration.json; restart the viewer to use them

Without calibration.json the hand-measured values of the printed adult jaw are used (DEFAULT).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .jaw import JAW_SCALE, LOWER

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.json"

REFERENCE_POINTS = (
    "incisal edge between 31 and 41 (on the tooth)",
    "tooth 36, lingual gum line (patient left)",
    "tooth 46, lingual gum line (patient right)",
)

# Printed adult jaw, measured by hand in the lower STL frame (mm): the lingual gum line of the
# last molars is 22 mm beside the midline, 42 mm behind the incisal edge and 9 mm below it.
DEFAULT_HALF_WIDTH_MM = 22.0
DEFAULT_ARCH_DEPTH_MM = 42.0
DEFAULT_GUM_LINE_BELOW_MM = 9.0


@dataclass(frozen=True)
class JawCalibration:
    """The three reference points in lower-jaw STL coordinates (mm)."""

    incisal: np.ndarray
    tooth_36: np.ndarray
    tooth_46: np.ndarray

    @classmethod
    def default(cls) -> "JawCalibration":
        """Values of the printed jaw; the incisal edge is read from the STL bounds."""
        bounds = LOWER.bounds_mm
        incisal = np.array([(bounds[0, 0] + bounds[1, 0]) / 2, LOWER.tip_height_mm, LOWER.incisor_z_mm])
        molar = np.array([DEFAULT_HALF_WIDTH_MM, -DEFAULT_GUM_LINE_BELOW_MM, DEFAULT_ARCH_DEPTH_MM]) * JAW_SCALE
        return cls(incisal, incisal + molar, incisal + molar * (-1, 1, 1))

    @classmethod
    def load_or_default(cls, path: Path = CALIBRATION_FILE) -> "JawCalibration":
        """The saved calibration, or the printed jaw's values if there is none or it is unusable."""
        if not path.exists():
            return cls.default()
        points = json.loads(path.read_text())["points"]
        calibration = cls(*(np.array(points[name]["lower_stl_mm"]) for name in REFERENCE_POINTS))
        if calibration.usable():
            return calibration
        print(f"{path.name}: the three points do not form a jaw (e.g. recorded at one spot); using the default jaw")
        return cls.default()

    def usable(self) -> bool:
        """True when the molar points lie on opposite sides and behind the incisal edge."""
        left = self.tooth_36 - self.incisal
        right = self.tooth_46 - self.incisal
        return left[0] * right[0] < 0 and left[2] > 0 and right[2] > 0


class CalibrationRecorder:
    """Walks through REFERENCE_POINTS, one Enter press per point."""

    def __init__(self):
        self.active = False
        self.recorded: dict[str, np.ndarray] = {}
        self.last_saved: Path | None = None
        self.discarded = False

    def press_enter(self, marker_world: np.ndarray) -> str | None:
        """One-button flow: start calibration if idle, else record the requested point (see `record`)."""
        if not self.active:
            self.start()
            return None
        return self.record(marker_world)

    def start(self) -> None:
        """(Re)start calibration: point 1 is requested, anything recorded before is dropped."""
        self.active = True
        self.recorded = {}
        self.discarded = False

    def record(self, marker_world: np.ndarray) -> str | None:
        """Record the marker position for the requested point; returns its name, None when not calibrating.

        A recording whose points do not form a jaw (e.g. the button pressed three times at one
        spot) is discarded, so an existing good calibration.json is never overwritten by it.
        """
        if not self.active:
            return None
        name = self.current_point()
        self.recorded[name] = marker_world.copy()
        if len(self.recorded) == len(REFERENCE_POINTS):
            self.active = False
            calibration = JawCalibration(*(LOWER.to_stl(self.recorded[point]) for point in REFERENCE_POINTS))
            if calibration.usable():
                self.last_saved = self._save()
            else:
                self.discarded = True
        return name

    def cancel(self) -> None:
        """Leave calibration mode; the points recorded so far are dropped."""
        self.active = False
        self.recorded = {}

    def current_point(self) -> str:
        return REFERENCE_POINTS[len(self.recorded)]

    def status(self) -> str:
        """One line for the on-screen text."""
        if self.active:
            step = len(self.recorded) + 1
            return f"CALIBRATION {step}/{len(REFERENCE_POINTS)}: put marker on {self.current_point()}, press Enter"
        if self.discarded:
            return "calibration DISCARDED: the three points do not form a jaw. Enter = start again"
        if self.last_saved:
            return f"saved {self.last_saved.name}; restart the viewer to use it. Enter = start again"
        return "Enter = start calibration"

    def _save(self) -> Path:
        points = {
            name: {"world_mm": _mm(point * 1000), "lower_stl_mm": _mm(LOWER.to_stl(point))}
            for name, point in self.recorded.items()
        }
        CALIBRATION_FILE.write_text(json.dumps({"recorded_at": datetime.now().isoformat(timespec="seconds"), "points": points}, indent=2))
        return CALIBRATION_FILE


def _mm(vector: np.ndarray) -> list[float]:
    return [round(float(v), 2) for v in vector]
