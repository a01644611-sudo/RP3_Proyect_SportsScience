from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageTk
import csv
import itertools
import json
import math
import os
import queue
import random
import socket
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import font as tkfont

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # must be set before pyplot-related imports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from dataclasses import dataclass

from datetime import datetime

BACKGROUND = "#0B1F2E"
PANEL = "#12324A"
PANEL_EDGE = "#1E4763"
GRID = "#1B4059"
CURVE = "#F2B441"      # this stroke
GHOST = "#5B86A5"      # previous stroke, for comparison
PEAK = "#7EE0C6"       # peak force marker and its vertical line
A70_COLOR = "#B48CE0"  # A70 marker, kept visually distinct from peak
TEXT = "#EDF4F9"
MUTED = "#8FB0C7"
OK = "#7EE0C6"         # good rhythm / drive phase-independent "good" colour
WARN = "#F2B441"       # drive phase colour, and mild feedback
BAD = "#E8715A"        # errors, session id missing, "too long/short"
RECOVERY_COLOR = "#6FA8DC"  # recovery phase colour, distinct from WARN/OK

BASE_DIR = Path(__file__).resolve().parent


def _find_assets_dir() -> Path:
    """Folder that holds the rowing images. Normally BASE_DIR/assets, but an
    unzipped download can end up nested (e.g. assets/rp3/assets), so search."""
    default = BASE_DIR / "assets"
    if (default / "catch.png").exists():
        return default
    for hit in sorted(BASE_DIR.rglob("catch.png")):
        return hit.parent
    return default


ASSETS_DIR = _find_assets_dir()

APP_DATA_DIR = Path.home() / "RP3 Rowing Experiment"
LOGS_DIR = APP_DATA_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Column order of the CSV. The first 16 columns are identical to the original
# script; the extra columns at the end are new.
CSV_COLUMNS = [
    "timestampMs",
    "powerWatts",
    "paceSecondsPer500Meters",
    "strokesPerMinute",
    "driveTimeSeconds",
    "recoverTimeSeconds",
    "peakForceNewtons",
    "strokeLengthMeters",
    "peakForcePositionMeters",
    "relativePeakForcePositionFraction",
    "energyJoules",
    "dragFactor",
    "forceCurve",
    "peakTargetHit",
    "peakPositionError",
    "peakPositionErrorPercent",
    # --- new ---
    "strokeMeanForceNewtons",
    "baselineMeanForceNewtons",
    "forceInRange",
    "peakHitWithForceGate",
    "driveTimeErrorPercent",
    "driveTimeOnTarget",
]

TARGET_SPM = 20.0

DRIVE_FRACTION = 1 / 3
RECOVERY_FRACTION = 2 / 3

TRIAL_DURATION_SECONDS = 5*60
REST_DURATION_SECONDS = 2*60
WARMUP_DURATION_SECONDS = 2 * 60

# ---- Condition 1 (rhythm) -------------------------------------------------
DRIVE_TIME_TOLERANCE = 0.05     # +/-5 % of the target drive time counts as "on target"
COUNTDOWN_SECONDS = 3           # "3, 2, 1, ROW!" before the metronome and trial clock start
METRONOME_FRAME_MS = 8         # ~120 fps requested. Windows timers resolve ~15 ms, so
                                # in practice expect 60-120 fps depending on the OS.

# ---- Condition 2 (peak force) --------------------------------------------
# NOTE: the original file had MAX (0.38) < TARGET (0.42), which made the zone
# impossible. Zone is now 34-42 % with the target in the middle.
PEAK_POSITION_MIN = 0.34
PEAK_POSITION_TARGET = 0.38
PEAK_POSITION_MAX = 0.42
FORCE_TOLERANCE = 0.20          # stroke mean force must be within +/-20 % of baseline mean force
MIN_BASELINE_STROKES = 3        # below this the force check is skipped

# ---- Condition 3 (force curve) -------------------------------------------
# Optimal drive-phase handle-force curve, normalised to the stroke's own peak
# (y, % of peak) against drive position (x, %). Shape taken from the averaged
# force curves of skilled rowers described by Kleshnev:
#   - force rises fast after the catch: ~60 % of peak within the first ~10 %
#     of the drive (little "catch slip", early A70)
#   - one smooth, rounded peak at ~40 % of the drive (leg drive)
#   - a smooth, gradual decline through body swing and arms, no second hump
#   - handle force falls back to ~0 at the finish
# Sources: Kleshnev, V. (2016) The Biomechanics of Rowing, Crowood Press;
#   Kleshnev, V. (2005) Comparison of on-water rowing with its simulation on
#   Concept2 and RowPerfect machines, ISBS Proceedings 23;
#   Smith, R.M. & Spinks, W.L. (1995) J Sports Sci 13(5):377-385
#   (elite rowers: earlier, smoother, more consistent force application).
REF_X = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
REF_Y = [5, 35, 60, 75, 86, 97, 100, 96, 87, 74, 57, 35, 20, 5]
# Width of the gradient (tolerance, in % of peak) at each control point.
# Wider at the steep catch and finish, where small timing shifts change force a lot.
REF_SIGMA = [14, 15, 13, 11, 10, 9, 8, 8, 9, 10, 11, 13, 14, 14]

# Phases of the drive (name, start %, end %) - same split as the rowing images.
DRIVE_PHASES = [
    ("CATCH", 0, 20),
    ("LEGS", 20, 55),
    ("BODY SWING", 55, 80),
    ("ARMS", 80, 100),
]

# (hint if force too LOW in this phase, hint if too HIGH)
PHASE_HINTS = {
    "CATCH": ("FASTER CATCH", "SMOOTHER CATCH"),
    "LEGS": ("USE MORE LEGS", "USE LESS LEGS"),
    "BODY SWING": ("ENGAGE CORE", "SMOOTHER TRANSITION"),
    "ARMS": ("USE MORE ARMS", "USE LESS ARMS"),
}

# Double peak ("two humps"): force drops between two peaks instead of one smooth
# peak, usually because legs and body swing are not connected. A dip counts when
# it is at least this deep (in % of the stroke's own peak) below the lower of the
# two peaks around it. Smaller wiggles are treated as sensor noise.
DIP_MIN_DEPTH = 8.0
DIP_BAD_DEPTH = 15.0     # from this depth the phase turns red instead of orange
DIP_HINT = "SMOOTHER TRANSITION BETWEEN LEGS AND ARMS"

XS = np.linspace(0, 100, 201)


def _smooth(a, w=9):
    pad = w // 2
    return np.convolve(np.pad(a, pad, mode="edge"), np.ones(w) / w, mode="valid")


IDEAL = _smooth(np.interp(XS, REF_X, REF_Y))
SIGMA = np.interp(XS, REF_X, REF_SIGMA)

OPT_CMAP = LinearSegmentedColormap.from_list("optimal", [PANEL, "#1D5A63", OK])

# ---- Experiment structure --------------------------------------------------
CONDITION_LABELS = {
    "baseline_pre": "Baseline (pre)",
    "rhythm": "Rhythm",
    "peak_force": "Peak force",
    "force_curve": "Force curve",
    "baseline_post": "Baseline (post)",
}
MIDDLE_CONDITIONS = ["rhythm", "peak_force", "force_curve"]
ORDERS = list(itertools.permutations(MIDDLE_CONDITIONS))   # all 6 possible orders


def order_label(order) -> str:
    return "  →  ".join(CONDITION_LABELS[k] for k in order)


def pick_font(root: tk.Tk) -> str:
    """First available of a short preference list, so the window looks the
    same-ish on Windows, macOS and Linux without shipping a font file."""
    available = {name.lower() for name in tkfont.families(root)}
    for candidate in ("Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans", "Arial"):
        if candidate.lower() in available:
            return candidate
    return "TkDefaultFont"


def _mix(c1, c2, t):
    """Blend two palette colours (t=0 -> c1, t=1 -> c2). Used only for fades/glows."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# --------------------------------------------------------------------------
# Baseline statistics (collected during the first baseline)
# --------------------------------------------------------------------------
class BaselineStats:
    """Running averages of the baseline strokes: mean force over the drive,
    peak force and relative peak position."""

    def __init__(self):
        self.count = 0
        self._sum_mean_force = 0.0
        self._sum_peak_force = 0.0
        self._sum_peak_pos = 0.0

    def add(self, analysis: dict) -> None:
        mean_force = analysis.get("mean_force")
        peak_force = analysis.get("peak_force")
        peak_pos = analysis.get("peak_position_fraction")
        if mean_force is None or peak_force is None or peak_pos is None:
            return
        self._sum_mean_force += mean_force
        self._sum_peak_force += peak_force
        self._sum_peak_pos += peak_pos
        self.count += 1

    @property
    def ready(self) -> bool:
        return self.count >= MIN_BASELINE_STROKES

    @property
    def mean_force(self):
        return self._sum_mean_force / self.count if self.count else None

    @property
    def mean_peak_force(self):
        return self._sum_peak_force / self.count if self.count else None

    @property
    def mean_peak_position(self):
        return self._sum_peak_pos / self.count if self.count else None

    def to_dict(self) -> dict:
        return {
            "strokes": self.count,
            "meanForceNewtons": self.mean_force,
            "meanPeakForceNewtons": self.mean_peak_force,
            "meanPeakPositionFraction": self.mean_peak_position,
        }


class StrokeLogger:

    def __init__(self, path: str, baseline_stats: BaselineStats | None = None):
        self.path = path
        self.baseline_stats = baseline_stats
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(CSV_COLUMNS)
            self._file.flush()

    def write(self, packet: dict) -> None:

        packet = dict(packet)
        peak_position = packet.get("relativePeakForcePositionFraction")

        try:
            peak_position = float(peak_position)

            packet["peakTargetHit"] = int(
                PEAK_POSITION_MIN <= peak_position <= PEAK_POSITION_MAX)

            packet["peakPositionError"] = abs(
                peak_position - PEAK_POSITION_TARGET)

            packet["peakPositionErrorPercent"] = abs(
                peak_position - PEAK_POSITION_TARGET) * 100

        except (TypeError, ValueError):
            packet["peakTargetHit"] = ""
            packet["peakPositionError"] = ""
            packet["peakPositionErrorPercent"] = ""

        # --- force relative to baseline -------------------------------------
        stroke_mean = analyze_stroke(packet).get("mean_force")
        packet["strokeMeanForceNewtons"] = stroke_mean

        stats = self.baseline_stats
        baseline_mean = stats.mean_force if (stats is not None and stats.ready) else None
        packet["baselineMeanForceNewtons"] = baseline_mean

        force_ok = None
        if stroke_mean is not None and baseline_mean:
            force_ok = int(abs(stroke_mean / baseline_mean - 1) <= FORCE_TOLERANCE)
        packet["forceInRange"] = force_ok

        if force_ok is not None and packet["peakTargetHit"] in (0, 1):
            packet["peakHitWithForceGate"] = int(bool(packet["peakTargetHit"]) and bool(force_ok))
        else:
            packet["peakHitWithForceGate"] = None

        # --- drive time vs metronome target -----------------------------------
        target_drive = (60.0 / TARGET_SPM) * DRIVE_FRACTION
        try:
            error = (float(packet.get("driveTimeSeconds")) - target_drive) / target_drive
            packet["driveTimeErrorPercent"] = error * 100
            packet["driveTimeOnTarget"] = int(abs(error) <= DRIVE_TIME_TOLERANCE)
        except (TypeError, ValueError):
            packet["driveTimeErrorPercent"] = None
            packet["driveTimeOnTarget"] = None

        row = []
        for column in CSV_COLUMNS:
            value = packet.get(column)
            # forceCurve is a list, so store it as a JSON string in one cell.
            row.append(json.dumps(value) if column == "forceCurve" else value)
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class StrokeReader(threading.Thread):
    """Reads newline-delimited JSON from the RP3 and posts it to a queue.

    Messages placed on the queue:
        ("stroke", packet_dict)
        ("status", text, state)   where state is "ok" | "wait" | "error"
    """

    def __init__(self, host: str, port: int, out_queue: "queue.Queue", logger: StrokeLogger | None):
        super().__init__(daemon=True, name="StrokeReader")
        self.host = host
        self.port = port
        self.out = out_queue
        self.logger = logger
        self.stop_event = threading.Event()

    def _status(self, text: str, state: str = "ok") -> None:
        self.out.put(("status", text, state))

    def run(self) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                self._status(f"Connecting to {self.host}:{self.port}", "wait")
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    # A short recv timeout lets the thread notice stop_event
                    # promptly even when the rower is resting between pieces.
                    sock.settimeout(1.0)
                    self._status(f"Connected to {self.host}:{self.port}", "ok")
                    backoff = 1.0
                    self._read_forever(sock)
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self._status(f"No connection ({exc}). Retrying in {backoff:.0f} s", "error")
                self.stop_event.wait(backoff)  # still reacts to a window close
                backoff = min(backoff * 1.7, 10.0)

    def _read_forever(self, sock: socket.socket) -> None:
        """Buffer bytes and emit every complete line.

        The buffer stays as *bytes* rather than str: a single recv() can cut a
        multi-byte UTF-8 character in half, and decoding half a character
        raises. Splitting on b"\\n" first and decoding whole lines is safe.
        """
        buffer = b""

        while not self.stop_event.is_set():
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue  # simply no stroke finished in the last second
            if not chunk:
                raise ConnectionError("the RP3 closed the connection")

            buffer += chunk

            # One recv may hold a fragment, one packet, or several packets.
            # Loop until no complete line is left, and keep the remainder.
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    packet = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._status(f"Skipped a malformed packet ({exc})", "error")
                    continue

                if self.logger is not None:
                    try:
                        self.logger.write(packet)
                    except Exception as exc:
                        self._status(f"CSV write failed: {exc}", "error")

                self.out.put(("stroke", packet))

    def stop(self) -> None:
        self.stop_event.set()


class DemoReader(threading.Thread):
    """Generates strokes locally, with no socket at all (--demo).

    To test the real network path instead, run rp3_simulator.py in a second
    terminal and point the dashboard at 127.0.0.1.
    """

    def __init__(self, out_queue: "queue.Queue", logger: StrokeLogger | None, speed: float = 1.0):
        super().__init__(daemon=True, name="DemoReader")
        self.out = out_queue
        self.logger = logger
        self.speed = speed
        self.stop_event = threading.Event()

    def run(self) -> None:
        from rp3_simulator import make_stroke_packet

        self.out.put(("status", "Demo mode: simulated strokes, no RP3 connected", "wait"))
        rng = random.Random()
        start = time.time()
        while not self.stop_event.is_set():
            packet = make_stroke_packet(int((time.time() - start) * 1000), skill=0.6, rng=rng)
            if self.logger is not None:
                self.logger.write(packet)
            self.out.put(("stroke", packet))
            self.stop_event.wait(60.0 / packet["strokesPerMinute"] / max(self.speed, 0.1))

    def stop(self) -> None:
        self.stop_event.set()


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def format_pace(seconds_per_500m) -> str:
    """Seconds per 500 m as m:ss.s, which is how rowers read pace."""
    try:
        seconds = float(seconds_per_500m)
    except (TypeError, ValueError):
        return "–"
    if seconds <= 0 or seconds > 3600:
        return "–"
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - minutes * 60:04.1f}"


def number(value, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "–"


def percent(fraction, decimals: int = 0) -> str:
    """0.317 -> '32%'."""
    try:
        return f"{float(fraction) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "–"


def parse_force_curve(raw, stroke_length_m: float):
    """Return (positions_m, forces_N) from the packet's forceCurve.

    Tolerant on purpose: the field may arrive as a list of dicts, as a JSON
    string (if it was read back from the CSV), or as bare force values with no
    positions, in which case the samples are spread evenly over the drive.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [], []
    if not raw:
        return [], []
    positions, forces = [], []
    for point in raw:
        if isinstance(point, dict):
            position = point.get("strokeLengthMeters", point.get("x"))
            force = point.get("forceNewtons", point.get("y"))
        else:
            position, force = None, point
        if force is None:
            continue
        positions.append(position)
        forces.append(float(force))

    if any(p is None for p in positions):
        span = stroke_length_m if stroke_length_m and stroke_length_m > 0 else 1.0
        divisor = max(len(forces) - 1, 1)
        positions = [span * i / divisor for i in range(len(forces))]
    else:
        positions = [float(p) for p in positions]

    return positions, forces


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze_stroke(packet: dict) -> dict:

    analysis = {}

    # 1. TIMING / RHYTHM
    drive = packet.get("driveTimeSeconds")
    recovery = packet.get("recoverTimeSeconds")

    try:
        drive = float(drive)
        recovery = float(recovery)

        cycle_time = drive + recovery

        analysis["cycle_time"] = cycle_time
        analysis["drive_fraction"] = drive / cycle_time
        analysis["recovery_drive_ratio"] = recovery / drive
        analysis["calculated_spm"] = 60 / cycle_time

    except (TypeError, ValueError, ZeroDivisionError):
        analysis["cycle_time"] = None
        analysis["drive_fraction"] = None
        analysis["recovery_drive_ratio"] = None
        analysis["calculated_spm"] = None

    # 2. FORCE CURVE
    stroke_length = float(packet.get("strokeLengthMeters") or 0)

    positions, forces = parse_force_curve(
        packet.get("forceCurve"),
        stroke_length
    )

    analysis["a70"] = None
    analysis["peak_position_fraction"] = None
    analysis["peak_force"] = None
    analysis["peak_position_m"] = None
    analysis["mean_force"] = None

    if len(positions) >= 2 and len(forces) >= 2:

        # Find actual maximum force
        peak_index = max(range(len(forces)), key=lambda i: forces[i])

        peak_force = forces[peak_index]
        peak_position = positions[peak_index]

        analysis["peak_force"] = peak_force
        analysis["peak_position_m"] = peak_position

        # 3. RELATIVE PEAK POSITION
        curve_start = positions[0]
        curve_end = positions[-1]
        curve_length = curve_end - curve_start

        if curve_length > 0:

            analysis["peak_position_fraction"] = (
                (peak_position - curve_start) / curve_length
            )

            # Mean force over the drive (area under the curve / drive length)
            area = 0.0
            for i in range(1, len(positions)):
                area += (positions[i] - positions[i - 1]) * (forces[i] + forces[i - 1]) / 2
            analysis["mean_force"] = area / curve_length

            # 4. A70: first point on the rising part where force reaches 70 % of peak
            force_70 = 0.70 * peak_force

            for i in range(1, peak_index + 1):

                if forces[i] >= force_70:

                    x1 = positions[i - 1]
                    x2 = positions[i]
                    y1 = forces[i - 1]
                    y2 = forces[i]

                    if y2 != y1:
                        fraction_between = (force_70 - y1) / (y2 - y1)
                        x70 = x1 + fraction_between * (x2 - x1)
                    else:
                        x70 = x2

                    analysis["a70"] = (x70 - curve_start) / curve_length
                    analysis["a70_position_m"] = x70

                    break

    # 5. VALUES ALREADY PROVIDED BY RP3
    analysis["rp3_spm"] = packet.get("strokesPerMinute")
    analysis["stroke_length"] = packet.get("strokeLengthMeters")
    analysis["energy"] = packet.get("energyJoules")

    return analysis


def evaluate_force_curve(positions, forces):
    """Compare a stroke with the optimal-curve template.

    Returns None if the curve is unusable, else a dict with:
      x, y     : curve normalised to drive position (%) and % of own peak
      score    : 0..1 overall closeness to the optimal curve
      phases   : list of {name, score, deviation, dip} per drive phase
                 (deviation < 0 means force is BELOW the optimal curve;
                 dip = depth of a double-peak dip in that phase, else 0)
      dip      : None, or {x, y, depth, phase} of the deepest dip between two peaks
    """
    pairs = sorted(zip(positions, forces))
    if len(pairs) < 2:
        return None

    pos = np.array([p[0] for p in pairs], dtype=float)
    frc = np.array([p[1] for p in pairs], dtype=float)

    span = pos[-1] - pos[0]
    peak = frc.max()
    if span <= 0 or peak <= 0:
        return None

    x = (pos - pos[0]) / span * 100.0
    y = frc / peak * 100.0

    resampled = np.interp(XS, x, y)
    optimality = np.exp(-0.5 * ((resampled - IDEAL) / SIGMA) ** 2)
    deviation = resampled - IDEAL

    dip = find_dip(resampled)

    phases = []
    for name, lo, hi in DRIVE_PHASES:
        mask = (XS >= lo) & (XS <= hi)
        part = optimality[mask]
        # Half mean, half worst 10 %: a short local error (like a dip) is no
        # longer averaged away by the good parts of the same phase.
        score = 0.5 * part.mean() + 0.5 * np.percentile(part, 10)
        dip_depth = 0.0
        if dip is not None and dip["phase"] == name:
            dip_depth = dip["depth"]
            score = min(score, 0.6 if dip_depth < DIP_BAD_DEPTH else 0.4)
        phases.append({
            "name": name,
            "score": float(score),
            "deviation": float(deviation[mask].mean()),
            "dip": dip_depth,
        })

    return {"x": x, "y": y, "score": float(optimality.mean()), "phases": phases, "dip": dip}


def find_dip(curve):
    """Deepest dip between two peaks of a resampled curve (on XS, % of peak).

    Returns None for a single smooth peak, else {x, y, depth, phase}. The
    phase is where the dip sits; a dip in the second half of the leg drive
    is a broken legs-to-body connection, so it is counted as BODY SWING.
    """
    s = _smooth(curve, 9)
    peaks = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1]]
    best = None
    for a, b in zip(peaks, peaks[1:]):
        low = a + int(np.argmin(s[a:b + 1]))
        depth = min(s[a], s[b]) - s[low]
        if depth >= DIP_MIN_DEPTH and (best is None or depth > best["depth"]):
            best = {"x": float(XS[low]), "y": float(s[low]), "depth": float(depth)}
    if best is None:
        return None

    dx = best["x"]
    if dx < 20:
        best["phase"] = "CATCH"
    elif dx < 45:
        best["phase"] = "LEGS"
    elif dx < 85:
        best["phase"] = "BODY SWING"
    else:
        best["phase"] = "ARMS"
    return best


class MetronomeTargets:
    def __init__(self):
        self.target_spm = TARGET_SPM
        self.target_drive_fraction = DRIVE_FRACTION
        self.target_recovery_fraction = RECOVERY_FRACTION

        self.cycle_time = 60.0 / self.target_spm
        self.target_drive_time = self.cycle_time * self.target_drive_fraction
        self.target_recovery_time = self.cycle_time * self.target_recovery_fraction


class MetronomeView(tk.Frame):

    TRACK_MARGIN = 70
    TRACK_Y = 380
    IMAGE_Y = 200

    DOT_RADIUS = 14

    def _load_image(self, path):
        image = Image.open(path)
        image = image.resize((400, 360), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def __init__(self, parent, family: str):

        super().__init__(
            parent,
            bg=PANEL,
            highlightbackground=PANEL_EDGE,
            highlightthickness=1,
            bd=0
        )

        self.family = family

        self.phase_label = tk.Label(
            self,
            text="READY",
            bg=PANEL,
            fg=MUTED,
            font=(family, 32, "bold")
        )
        self.phase_label.pack(pady=(4, 0))

        self.phase_images = {
            "catch": self._load_image(ASSETS_DIR / "catch.png"),
            "leg_extension": self._load_image(ASSETS_DIR / "leg_extension.png"),
            "trunk_swing": self._load_image(ASSETS_DIR / "trunk_swing.png"),
            "arm_pull": self._load_image(ASSETS_DIR / "arm_pull.png"),
            "arms_away": self._load_image(ASSETS_DIR / "arms_away.png"),
            "recovery_trunk": self._load_image(ASSETS_DIR / "recovery_trunk.png"),
            "leg_compression": self._load_image(ASSETS_DIR / "leg_compression.png")
        }

        self.canvas = tk.Canvas(
            self,
            bg=PANEL,
            highlightthickness=0,
            height=500
        )
        self.canvas.pack(fill="x", padx=40, pady=(4, 8))
        self.canvas.bind("<Configure>", self._on_resize)

        self._width = 0
        self._dot = None
        self._fill_line = None
        self._rower_item = None
        # caches so we only touch Tk when something actually changed (faster frames)
        self._last_phase = None
        self._last_key = None
        self._draw_static()

    def _on_resize(self, _event=None):
        self._draw_static()

    def _draw_static(self):

        self.canvas.delete("all")
        self._last_phase = None
        self._last_key = None

        width = max(self.canvas.winfo_width(), 300)
        self._width = width

        y = self.TRACK_Y
        left = self.TRACK_MARGIN
        right = width - self.TRACK_MARGIN

        # Track
        self.canvas.create_line(left, y, right, y, fill=GRID, width=10, capstyle=tk.ROUND)

        # Labels
        self.canvas.create_text(left, y + 18, text="FINISH", fill=OK,
                                font=(self.family, 12), anchor="n")
        self.canvas.create_text(right, y + 18, text="CATCH", fill=OK,
                                font=(self.family, 12), anchor="n")

        # Rower image
        self._rower_item = self.canvas.create_image(
            width / 2, self.IMAGE_Y, image=self.phase_images["catch"])

        # Progress fill and dot are drawn last so they sit on top of the track
        self._fill_line = self.canvas.create_line(
            right, y, right, y, fill=WARN, width=10, capstyle=tk.ROUND)

        r = self.DOT_RADIUS
        self._dot = self.canvas.create_oval(
            right - r, y - r, right + r, y + r,
            fill=MUTED, outline=PANEL, width=2)

    def set_phase(self, phase: str, fraction: float):

        fraction = min(max(fraction, 0.0), 1.0)
        colour = WARN if phase == "drive" else RECOVERY_COLOR

        # Only update label/colours when the phase flips
        if phase != self._last_phase:
            self._last_phase = phase
            self.phase_label.config(text=phase.upper(), fg=colour)
            self.canvas.itemconfig(self._fill_line, fill=colour)
            self.canvas.itemconfig(self._dot, fill=colour)

        width = self._width or max(self.canvas.winfo_width(), 300)
        left = self.TRACK_MARGIN
        right = width - self.TRACK_MARGIN
        y = self.TRACK_Y

        if phase == "drive":
            x = right - fraction * (right - left)
            self.canvas.coords(self._fill_line, right, y, x, y)
        else:
            x = left + fraction * (right - left)
            self.canvas.coords(self._fill_line, left, y, x, y)

        r = self.DOT_RADIUS
        self.canvas.coords(self._dot, x - r, y - r, x + r, y + r)

        self.update_rowing_image(phase, fraction)

    def set_idle(self):
        self.phase_label.config(text="READY", fg=MUTED)

        width = self._width or max(self.canvas.winfo_width(), 300)
        right = width - self.TRACK_MARGIN
        y = self.TRACK_Y

        self.canvas.coords(self._fill_line, right, y, right, y)
        self.canvas.itemconfig(self._fill_line, fill=MUTED)

        r = self.DOT_RADIUS
        self.canvas.coords(self._dot, right - r, y - r, right + r, y + r)
        self.canvas.itemconfig(self._dot, fill=MUTED)

        self._last_phase = None
        self._last_key = "catch"
        self.canvas.itemconfig(self._rower_item, image=self.phase_images["catch"])

    def update_rowing_image(self, phase, fraction):

        if phase == "drive":
            if fraction < 0.20:
                key = "catch"
            elif fraction < 0.55:
                key = "leg_extension"
            elif fraction < 0.80:
                key = "trunk_swing"
            else:
                key = "arm_pull"
        else:
            if fraction < 0.30:
                key = "arms_away"
            elif fraction < 0.60:
                key = "recovery_trunk"
            else:
                key = "leg_compression"

        if key != self._last_key:
            self._last_key = key
            self.canvas.itemconfig(self._rower_item, image=self.phase_images[key])


@dataclass
class ExperimentConfig:
    participant_id: str
    session_id: str
    port: int
    order: list   # order of the 3 middle conditions, e.g. ["peak_force", "rhythm", "force_curve"]


class ExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RP3 Rowing Experiment")
        self.geometry("1180x900")
        self.configure(bg=BACKGROUND)
        self.minsize(900, 760)

        self.family = pick_font(self)

        self.current_frame = None
        self.reader = None
        self.logger = None
        self.experiment_config = None   # (was self.config, which shadowed Tk.config)

        self.data_queue = queue.Queue()

        self.sequence = []
        self.baseline_stats = BaselineStats()
        self.current_condition_index = 0
        self.experiment_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.show_setup()

    # ------------------------------------------------------------ screens
    def clear_screen(self):
        if self.current_frame is not None:
            leave = getattr(self.current_frame, "on_leave", None)
            if leave is not None:
                leave()
            self.current_frame.destroy()
            self.current_frame = None

    def _show(self, frame):
        self.clear_screen()
        self.current_frame = frame
        self.current_frame.pack(fill="both", expand=True)

    def show_setup(self):
        self._show(SetupFrame(self, self))

    def show_rest(self):
        next_name = self.sequence[self.current_condition_index]
        self._show(RestFrame(self, self, CONDITION_LABELS[next_name]))

    def show_finished(self):
        self._show(FinishedFrame(self, self))

    # --------------------------------------------------------- experiment flow
    def start_experiment(self, config):
        self.experiment_config = config
        self.sequence = ["baseline_pre", *config.order, "baseline_post"]
        self.baseline_stats = BaselineStats()
        self.current_condition_index = 0
        self.experiment_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("Participant:", config.participant_id)
        print("Session:", config.session_id)
        print("Port:", config.port)
        print("Sequence:", self.sequence)

        self._save_meta()
        self.run_condition(0)

    def run_condition(self, index):
        name = self.sequence[index]

        if name == "baseline_pre":
            frame = BaselineFrame(self, self, title="BASELINE", record_baseline=True)
        elif name == "baseline_post":
            frame = BaselineFrame(self, self, title="BASELINE (POST)", record_baseline=False)
        elif name == "rhythm":
            frame = RhythmFrame(self, self)
        elif name == "peak_force":
            frame = PeakForceFrame(self, self)
        else:
            frame = ForceCurveFrame(self, self)

        self._show(frame)
        self.start_condition_io(f"C{index + 1}_{name}", name)

    def start_next_condition(self):
        if self.current_condition_index >= len(self.sequence):
            self.show_finished()
            return
        try:
            self.run_condition(self.current_condition_index)
        except Exception as exc:
            # Without this the error only goes to the console and the button
            # just seems to do nothing.
            import traceback
            from tkinter import messagebox
            traceback.print_exc()
            messagebox.showerror("Could not start the next trial", f"{type(exc).__name__}: {exc}")

    def finish_condition(self):
        self.stop_condition_io()
        self.current_condition_index += 1
        self._save_meta()

        if self.current_condition_index >= len(self.sequence):
            self.show_finished()
        else:
            self.show_rest()

    def _save_meta(self):
        """Write the condition order and baseline statistics next to the CSVs."""
        cfg = self.experiment_config
        if cfg is None:
            return
        path = LOGS_DIR / f"{cfg.participant_id}_{cfg.session_id}_{self.experiment_stamp}_session.json"
        data = {
            "participant": cfg.participant_id,
            "session": cfg.session_id,
            "stamp": self.experiment_stamp,
            "sequence": self.sequence,
            "baseline": self.baseline_stats.to_dict(),
        }
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            print("Could not write session file:", exc)

    # ----------------------------------------------------------------- I/O
    def start_condition_io(self, condition_name, base_name=None):
        self.stop_condition_io()

        # Remove any packets left from a previous condition
        while True:
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        cfg = self.experiment_config
        filename = (
            f"{cfg.participant_id}_"
            f"{cfg.session_id}_"
            f"{self.experiment_stamp}_"
            f"{condition_name}.csv"
        )
        path = LOGS_DIR / filename

        # The first baseline *creates* the stats, so it logs without them.
        stats = None if base_name == "baseline_pre" else self.baseline_stats
        self.logger = StrokeLogger(str(path), baseline_stats=stats)

        self.reader = StrokeReader(
            "localhost",
            cfg.port,
            self.data_queue,
            self.logger
        )
        self.reader.start()

        print("Started:", condition_name)
        print("Saving to:", path)

    def stop_condition_io(self):
        if self.reader is not None:
            self.reader.stop()
            self.reader.join(timeout=2.0)
            self.reader = None

        if self.logger is not None:
            self.logger.close()
            self.logger = None


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------
class FinishedFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BACKGROUND)

        self.app = app
        self.family = pick_font(app)

        card = tk.Frame(self, bg=PANEL, highlightbackground=PANEL_EDGE, highlightthickness=1)
        card.pack(expand=True, padx=150, pady=120, fill="both")

        tk.Label(card, text="EXPERIMENT COMPLETE", bg=PANEL, fg=OK,
                 font=(self.family, 32, "bold")).pack(pady=(100, 20))

        tk.Label(card, text="All five rowing conditions have been completed.",
                 bg=PANEL, fg=TEXT, font=(self.family, 14)).pack(pady=(0, 15))

        tk.Label(card, text="Thank you for participating.",
                 bg=PANEL, fg=MUTED, font=(self.family, 12)).pack(pady=(0, 35))

        tk.Button(card, text="Close", command=self.app.destroy,
                  bg=OK, fg=BACKGROUND, font=(self.family, 12, "bold"),
                  relief="flat", padx=22, pady=10).pack()


class RestFrame(tk.Frame):
    def __init__(self, parent, app, next_label=""):
        super().__init__(parent, bg=BACKGROUND)

        self.app = app
        self.family = pick_font(app)
        self.next_label = next_label

        self.remaining = REST_DURATION_SECONDS

        self._build_layout()
        self._update_timer()

    def _build_layout(self):
        card = tk.Frame(self, bg=PANEL, highlightbackground=PANEL_EDGE, highlightthickness=1)
        card.pack(expand=True, padx=150, pady=120, fill="both")

        tk.Label(card, text="REST", bg=PANEL, fg=TEXT,
                 font=(self.family, 36, "bold")).pack(pady=(90, 20))

        tk.Label(card, text="Take a short break before the next trial.",
                 bg=PANEL, fg=MUTED, font=(self.family, 14)).pack(pady=(0, 10))

        if self.next_label:
            tk.Label(card, text=f"Next: {self.next_label}", bg=PANEL, fg=TEXT,
                     font=(self.family, 14, "bold")).pack(pady=(0, 20))

        self.timer_var = tk.StringVar(value="00:00")

        tk.Label(card, textvariable=self.timer_var, bg=PANEL, fg=OK,
                 font=(self.family, 42, "bold")).pack(pady=20)

        self.next_button = tk.Button(
            card,
            text="Start next trial",
            command=self.app.start_next_condition,
            bg=OK,
            fg=BACKGROUND,
            font=(self.family, 12, "bold"),
            relief="flat",
            padx=22,
            pady=10,
            state="disabled"
        )
        self.next_button.pack(pady=(35, 20))

    def _update_timer(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        minutes = self.remaining // 60
        seconds = self.remaining % 60

        self.timer_var.set(f"{minutes:02d}:{seconds:02d}")

        if self.remaining <= 0:
            self.next_button.config(state="normal")
            return

        self.remaining -= 1
        self.after(1000, self._update_timer)


class SetupFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BACKGROUND)

        self.app = app
        self.family = pick_font(app)

        card = tk.Frame(self, bg=PANEL, highlightbackground=PANEL_EDGE, highlightthickness=1)
        card.pack(expand=True, padx=100, pady=60)

        tk.Label(card, text="RP3 ROWING EXPERIMENT", bg=PANEL, fg=TEXT,
                 font=(self.family, 24, "bold")).pack(padx=80, pady=(35, 25))

        tk.Label(card, text="Participant ID", bg=PANEL, fg=MUTED,
                 font=(self.family, 9)).pack(anchor="w", padx=60)

        self.participant_entry = tk.Entry(card, font=(self.family, 12), bg=BACKGROUND,
                                          fg=TEXT, insertbackground=TEXT, relief="flat")
        self.participant_entry.pack(fill="x", padx=60, pady=(5, 14), ipady=7)

        tk.Label(card, text="Session ID", bg=PANEL, fg=MUTED,
                 font=(self.family, 9)).pack(anchor="w", padx=60)

        self.session_entry = tk.Entry(card, font=(self.family, 12), bg=BACKGROUND,
                                      fg=TEXT, insertbackground=TEXT, relief="flat")
        self.session_entry.pack(fill="x", padx=60, pady=(5, 14), ipady=7)

        tk.Label(card, text="RP3 Port", bg=PANEL, fg=MUTED,
                 font=(self.family, 9)).pack(anchor="w", padx=60)

        self.port_entry = tk.Entry(card, font=(self.family, 12), bg=BACKGROUND,
                                   fg=TEXT, insertbackground=TEXT, relief="flat")
        self.port_entry.insert(0, "58382")
        self.port_entry.pack(fill="x", padx=60, pady=(5, 14), ipady=7)

        # ---- order of the three conditions ----
        tk.Label(card, text="Order of conditions 2-4 (baselines are always first and last)",
                 bg=PANEL, fg=MUTED, font=(self.family, 9)).pack(anchor="w", padx=60)

        self.order_labels = {order_label(o): list(o) for o in ORDERS}
        self.order_var = tk.StringVar(value=order_label(ORDERS[0]))

        menu = tk.OptionMenu(card, self.order_var, *self.order_labels.keys())
        menu.config(bg=BACKGROUND, fg=TEXT, activebackground=PANEL_EDGE,
                    activeforeground=TEXT, relief="flat", highlightthickness=0,
                    font=(self.family, 11), anchor="w")
        menu["menu"].config(bg=BACKGROUND, fg=TEXT, activebackground=PANEL_EDGE,
                            activeforeground=TEXT, font=(self.family, 11))
        menu.pack(fill="x", padx=60, pady=(5, 22), ipady=4)

        tk.Button(card, text="Start experiment", command=self.submit,
                  bg=OK, fg=BACKGROUND, font=(self.family, 11, "bold"),
                  relief="flat", padx=20, pady=10).pack(pady=(5, 35))

    def submit(self):
        participant = self.participant_entry.get().strip()
        session = self.session_entry.get().strip()

        if not participant or not session:
            return

        try:
            port = int(self.port_entry.get())
        except ValueError:
            return

        config = ExperimentConfig(
            participant_id=participant,
            session_id=session,
            port=port,
            order=self.order_labels[self.order_var.get()]
        )

        self.app.start_experiment(config)


# --------------------------------------------------------------------------
# Trial base class: shared timer + data queue handling
# --------------------------------------------------------------------------
class TrialFrame(tk.Frame):
    """Common behaviour of all trial screens.

    - 30 s trial timer that ends the condition
    - drains the stroke queue and calls handle_stroke() for every stroke
    - every callback checks that the frame still exists, so a trial that was
      ended with the button can't be "finished" a second time by its old timer
    """

    def __init__(self, parent, app):
        super().__init__(parent, bg=BACKGROUND)
        self.app = app
        self.family = pick_font(app)
        self.queue = app.data_queue
        self.stroke_count = 0
        self.trial_start_time = time.perf_counter()
        self.timer_after_id = None
        self.timer_var = tk.StringVar(
            value=f"{TRIAL_DURATION_SECONDS // 60:02d}:{TRIAL_DURATION_SECONDS % 60:02d}")
        # rower's current stroke rate, updated on every stroke (shown with show_spm)
        self.live_spm_var = tk.StringVar(value="– spm")

    def _alive(self) -> bool:
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    def start_loops(self, start_clock: bool = True):
        self.after(40, self._drain_queue)
        if start_clock:
            self.start_trial_clock()

    def start_trial_clock(self):
        """(Re)start the 30 s trial timer from now."""
        self.trial_start_time = time.perf_counter()
        self._update_trial_timer()

    def _top_bar(self, parent, title, right_text=None, pady=(0, 20), show_spm=False):
        top = tk.Frame(parent, bg=BACKGROUND)
        top.pack(fill="x", pady=pady)

        if show_spm:
            # far right corner: the rower's current stroke rate
            tk.Label(top, textvariable=self.live_spm_var, bg=BACKGROUND, fg=TEXT,
                     font=(self.family, 22, "bold")).pack(side="right")
        tk.Label(top, textvariable=self.timer_var, bg=BACKGROUND, fg=OK,
                 font=(self.family, 16, "bold")).pack(side="right", padx=(0, 20))
        tk.Label(top, text=title, bg=BACKGROUND, fg=TEXT,
                 font=(self.family, 26, "bold")).pack(side="left")
        if right_text:
            tk.Label(top, text=right_text, bg=BACKGROUND, fg=MUTED,
                     font=(self.family, 14, "bold")).pack(side="right")
        return top

    def _drain_queue(self):
        if not self._alive():
            return

        while True:
            try:
                message = self.queue.get_nowait()
            except queue.Empty:
                break

            if message[0] == "stroke":
                self.stroke_count += 1
                try:
                    self.live_spm_var.set(f"{float(message[1].get('strokesPerMinute')):.1f} spm")
                except (TypeError, ValueError):
                    pass
                self.handle_stroke(message[1])

        self.after(40, self._drain_queue)

    def handle_stroke(self, packet):
        pass

    def _update_trial_timer(self):
        if not self._alive():
            return

        elapsed = time.perf_counter() - self.trial_start_time
        remaining = max(0, TRIAL_DURATION_SECONDS - int(elapsed))

        self.timer_var.set(f"{remaining // 60:02d}:{remaining % 60:02d}")

        if elapsed >= TRIAL_DURATION_SECONDS:
            self.app.finish_condition()
            return

        self.timer_after_id = self.after(200, self._update_trial_timer)


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
class BaselineFrame(TrialFrame):
    def __init__(self, parent, app, title="BASELINE", record_baseline=False):
        super().__init__(parent, app)

        self.title_text = title
        self.record_baseline = record_baseline

        self.power_var = tk.StringVar(value="–")
        self.pace_var = tk.StringVar(value="–")
        self.spm_var = tk.StringVar(value="–")
        self.distance_var = tk.StringVar(value="0 m")
        self.time_var = tk.StringVar(value="00:00")
        self.strokes_var = tk.StringVar(value="0")

        self._build_layout()
        self.start_loops()
        self.after(200, self.update_time)

    def _build_layout(self):
        outer = tk.Frame(self, bg=BACKGROUND)
        outer.pack(fill="both", expand=True, padx=30, pady=25)

        self._top_bar(outer, self.title_text, f"Target rate: {TARGET_SPM:.0f} spm")

        grid = tk.Frame(outer, bg=BACKGROUND)
        grid.pack(fill="both", expand=True)

        for c in range(3):
            grid.columnconfigure(c, weight=1)

        for r in range(2):
            grid.rowconfigure(r, weight=1)

        self._metric_card(grid, 0, 0, "POWER", self.power_var, "W")
        self._metric_card(grid, 0, 1, "500 M PACE", self.pace_var, "")
        self._metric_card(grid, 0, 2, "STROKE RATE", self.spm_var, "spm")

        self._metric_card(grid,1,0,"DISTANCE",self.distance_var,"")
        self._metric_card(grid, 1, 1, "TIME", self.time_var, "")
        self._metric_card(grid, 1, 2, "TOTAL STROKES", self.strokes_var, "")

        tk.Button(outer, text="Finish Trial", command=self.app.finish_condition,
                  bg=OK, fg=BACKGROUND, font=(self.family, 11, "bold"),
                  relief="flat", padx=18, pady=9).pack(pady=(15, 0))

    def _metric_card(self, parent, row, column, title, var, unit):
        card = tk.Frame(parent, bg=PANEL, highlightbackground=PANEL_EDGE, highlightthickness=1)
        card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)

        tk.Label(card, text=title, bg=PANEL, fg=MUTED,
                 font=(self.family, 11, "bold")).pack(pady=(35, 8))

        value_row = tk.Frame(card, bg=PANEL)
        value_row.pack(expand=True)

        tk.Label(value_row, textvariable=var, bg=PANEL, fg=TEXT,
                 font=(self.family, 36, "bold")).pack(side="left")

        if unit:
            tk.Label(value_row, text=unit, bg=PANEL, fg=MUTED,
                     font=(self.family, 14)).pack(side="left", padx=(6, 0), pady=(14, 0))

    def handle_stroke(self, packet):
        # Collect baseline statistics from EVERY stroke of the first baseline
        if self.record_baseline:
            self.app.baseline_stats.add(analyze_stroke(packet))

        power = packet.get("powerWatts")
        pace = packet.get("paceSecondsPer500Meters")
        spm = packet.get("strokesPerMinute")

        self.power_var.set(f"{float(power):.0f}" if power is not None else "–")
        self.pace_var.set(format_pace(pace))
        self.spm_var.set(f"{float(spm):.1f}" if spm is not None else "–")
        self.strokes_var.set(str(self.stroke_count))

    def update_time(self):
        if not self._alive():
            return

        elapsed = int(time.perf_counter() - self.trial_start_time)
        self.time_var.set(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

        self.after(200, self.update_time)


# --------------------------------------------------------------------------
# Condition: rhythm
# --------------------------------------------------------------------------
class RhythmFrame(TrialFrame):
    def __init__(self, parent, app):
        super().__init__(parent, app)

        self.targets = MetronomeTargets()
        self.metronome_running = False
        self.metronome_start_time = 0.0
        self.metronome_after_id = None

        self._build_layout()
        # The trial clock and metronome start after the countdown.
        self.start_loops(start_clock=False)
        self.after(500, self._countdown, COUNTDOWN_SECONDS)

    def _build_layout(self):
        outer = tk.Frame(self, bg=BACKGROUND)
        outer.pack(fill="both", expand=True, padx=30, pady=15)

        self._top_bar(outer, "RHYTHM AND SEQUENCING", f"Target {TARGET_SPM:.0f} spm", pady=(0, 6), show_spm=True)

        # Compact feedback strip: leaves the metronome and images large
        strip = tk.Frame(outer, bg=BACKGROUND)
        strip.pack(fill="x")

        self.feedback_var = tk.StringVar(value="GET READY")
        self.feedback_label = tk.Label(
            strip, textvariable=self.feedback_var, bg=BACKGROUND, fg=MUTED,
            font=(self.family, 24, "bold"))
        self.feedback_label.pack()

        self.detail_var = tk.StringVar(
            value=f"Target drive {self.targets.target_drive_time:.2f} s  "
                  f"(±{DRIVE_TIME_TOLERANCE * 100:.0f}%)")
        tk.Label(strip, textvariable=self.detail_var, bg=BACKGROUND, fg=MUTED,
                 font=(self.family, 12)).pack(pady=(0, 6))

        self.metronome_view = MetronomeView(outer, self.family)
        self.metronome_view.pack(fill="both", expand=True, pady=(4, 10))

        controls = tk.Frame(outer, bg=BACKGROUND)
        controls.pack(fill="x")

        tk.Button(controls, text="Finish Trial", command=self.app.finish_condition,
                  bg=WARN, fg=BACKGROUND, font=(self.family, 11, "bold"),
                  relief="flat", padx=18, pady=9).pack(side="right")

    # ---------------------------------------------------------- feedback
    def handle_stroke(self, packet):
        if not self.metronome_running:
            return      # still counting down: strokes are logged but get no feedback

        try:
            drive = float(packet.get("driveTimeSeconds"))
        except (TypeError, ValueError):
            return

        target = self.targets.target_drive_time
        error = (drive - target) / target          # +0.10 = drive 10 % too long

        if abs(error) <= DRIVE_TIME_TOLERANCE:
            msg, colour = "GOOD", OK
        else:
            # drive shorter than target = too quick, longer = too slow
            msg = "DRIVE TOO SLOW" if error > 0 else "DRIVE TOO QUICK"
            colour = WARN if abs(error) <= 2 * DRIVE_TIME_TOLERANCE else BAD

        self.feedback_var.set(msg)
        self.feedback_label.config(fg=colour)
        self.detail_var.set(f"Drive {drive:.2f} s  |  target {target:.2f} s")

    # --------------------------------------------------------- countdown
    def _countdown(self, remaining):
        if not self._alive():
            return
        label = self.metronome_view.phase_label
        if remaining > 0:
            label.config(text=str(remaining), fg=TEXT)
            self.after(1000, self._countdown, remaining - 1)
        else:
            label.config(text="ROW!", fg=OK)
            self.feedback_var.set("ROW TO THE METRONOME")
            self.start_trial_clock()
            self.start_metronome()

    # --------------------------------------------------------- metronome
    def start_metronome(self):
        if not self._alive():
            return
        self.metronome_running = True
        self.metronome_start_time = time.perf_counter()

        self.update_metronome()

    def update_metronome(self):
        if not self.metronome_running or not self._alive():
            return

        elapsed = time.perf_counter() - self.metronome_start_time
        position_in_cycle = elapsed % self.targets.cycle_time

        if position_in_cycle < self.targets.target_drive_time:
            phase = "drive"
            fraction = position_in_cycle / self.targets.target_drive_time

        else:
            phase = "recovery"
            recovery_elapsed = position_in_cycle - self.targets.target_drive_time
            fraction = recovery_elapsed / self.targets.target_recovery_time

        self.metronome_view.set_phase(phase, fraction)

        self.metronome_after_id = self.after(METRONOME_FRAME_MS, self.update_metronome)

    def stop_metronome(self):
        self.metronome_running = False

        if self.metronome_after_id is not None:
            try:
                self.after_cancel(self.metronome_after_id)
            except tk.TclError:
                pass

            self.metronome_after_id = None

        self.metronome_view.set_idle()

    def on_leave(self):
        self.stop_metronome()


# --------------------------------------------------------------------------
# Condition: peak force
# --------------------------------------------------------------------------
class PeakForceFrame(TrialFrame):
    """2-D target: x = relative peak position, y = stroke force (mean drive force, N)
    compared with the baseline. A stroke is on target when both are inside the box."""

    FRAME_MS = 33
    PAD_L, PAD_R, PAD_T, PAD_B = 80, 40, 30, 60

    def __init__(self, parent, app):
        super().__init__(parent, app)

        self.max_force_seen = 0.0

        # marker eases toward the latest stroke
        self.marker_pos = 0.0
        self.marker_target = 0.0
        self.marker_force = None          # N, None until the first stroke
        self.marker_force_target = None
        self.marker_colour = MUTED

        self._build_layout()
        self.start_loops()
        self.after(self.FRAME_MS, self._animate)

    # ------------------------------------------------------------------ layout
    def _build_layout(self):
        outer = tk.Frame(self, bg=BACKGROUND)
        outer.pack(fill="both", expand=True, padx=35, pady=25)

        self._top_bar(outer, "PEAK FORCE CONTROL", f"Target {TARGET_SPM:.0f} spm", pady=0, show_spm=True)

        self.feedback_var = tk.StringVar(value="Take a stroke to start")
        self.feedback_label = tk.Label(
            outer, textvariable=self.feedback_var, bg=BACKGROUND, fg=MUTED,
            font=(self.family, 26, "bold"))
        self.feedback_label.pack(pady=(16, 2))

        stats = self.app.baseline_stats
        if stats.ready:
            lo, hi = self._force_band()
            info = f"Peak at {PEAK_POSITION_MIN * 100:.0f}–{PEAK_POSITION_MAX * 100:.0f}%  |  force {lo:.0f}–{hi:.0f} N"
        else:
            info = "No baseline force available: judging peak position only"
        tk.Label(outer, text=info, bg=BACKGROUND, fg=MUTED,
                 font=(self.family, 12)).pack(pady=(0, 4))

        self.game_canvas = tk.Canvas(
            outer, bg=PANEL, highlightthickness=2,
            highlightbackground=PANEL_EDGE, height=420)
        self.game_canvas.pack(fill="both", expand=True, padx=40, pady=8)

        controls = tk.Frame(outer, bg=BACKGROUND)
        controls.pack(fill="x")

        tk.Button(controls, text="Finish Trial", command=self.app.finish_condition,
                  bg=PEAK, fg=BACKGROUND, font=(self.family, 11, "bold"),
                  relief="flat", padx=18, pady=9).pack(side="right")

    # --------------------------------------------------------------- geometry
    def _force_band(self):
        """Accepted stroke-force range (N), or None without a baseline."""
        stats = self.app.baseline_stats
        if not (stats.ready and stats.mean_force):
            return None
        return (stats.mean_force * (1 - FORCE_TOLERANCE),
                stats.mean_force * (1 + FORCE_TOLERANCE))

    def _force_range(self):
        """Force (N) shown at the bottom and top of the plot."""
        stats = self.app.baseline_stats
        if stats.ready and stats.mean_force:
            return stats.mean_force * 0.4, stats.mean_force * 1.6
        return 0.0, max(400.0, self.max_force_seen * 1.3)

    def _geom(self):
        w = max(self.game_canvas.winfo_width(), 400)
        h = max(self.game_canvas.winfo_height(), 300)
        return w, h, self.PAD_L, w - self.PAD_R, self.PAD_T, h - self.PAD_B

    def _to_xy(self, pos, force):
        w, h, left, right, top, bottom = self._geom()
        f_lo, f_hi = self._force_range()
        x = left + min(max(pos, 0.0), 1.0) * (right - left)
        if force is None:
            y = (top + bottom) / 2
        else:
            frac = min(max((force - f_lo) / (f_hi - f_lo), 0.0), 1.0)
            y = bottom - frac * (bottom - top)
        return x, y

    # --------------------------------------------------------------- drawing
    def _animate(self):
        if not self._alive():
            return

        self.marker_pos += (self.marker_target - self.marker_pos) * 0.22
        if self.marker_force_target is not None:
            if self.marker_force is None:
                self.marker_force = self.marker_force_target
            self.marker_force += (self.marker_force_target - self.marker_force) * 0.22

        self._draw()
        self.after(self.FRAME_MS, self._animate)

    def _draw(self):
        c = self.game_canvas
        c.delete("all")

        w, h, left, right, top, bottom = self._geom()
        span = right - left
        f_lo, f_hi = self._force_range()

        # --- plot frame
        c.create_rectangle(left, top, right, bottom, outline=PANEL_EDGE, width=2)

        # --- x grid: peak position, catch -> finish
        for i in range(11):
            tx = left + span * i / 10
            c.create_line(tx, top, tx, bottom, fill=GRID, width=1)
            if i % 2 == 0:
                c.create_text(tx, bottom + 14, text=f"{i * 10}%", fill=MUTED,
                              font=(self.family, 10, "bold"))
        c.create_text(left, bottom + 34, text="CATCH", fill=MUTED, anchor="w",
                      font=(self.family, 10, "bold"))
        c.create_text((left + right) / 2, bottom + 34, text="PEAK POSITION",
                      fill=MUTED, font=(self.family, 10, "bold"))
        c.create_text(right, bottom + 34, text="FINISH", fill=MUTED, anchor="e",
                      font=(self.family, 10, "bold"))

        # --- y grid: stroke force (N)
        for i in range(5):
            force = f_lo + (f_hi - f_lo) * i / 4
            _, ty = self._to_xy(0, force)
            c.create_line(left, ty, right, ty, fill=GRID, width=1)
            c.create_text(left - 8, ty, text=f"{force:.0f} N", fill=MUTED,
                          anchor="e", font=(self.family, 10, "bold"))
        c.create_text(left + 6, top - 14, text="STROKE FORCE", fill=MUTED,
                      anchor="w", font=(self.family, 10, "bold"))

        # --- baseline reference line
        stats = self.app.baseline_stats
        band = self._force_band()
        if band:
            _, by = self._to_xy(0, stats.mean_force)
            c.create_line(left, by, right, by, fill=MUTED, dash=(6, 4), width=1)
            c.create_text(right - 6, by - 10, text="BASELINE", fill=MUTED,
                          anchor="e", font=(self.family, 9, "bold"))

        # --- target box (position zone x force band)
        tl = left + PEAK_POSITION_MIN * span
        tr = left + PEAK_POSITION_MAX * span
        if band:
            _, tt = self._to_xy(0, band[1])
            _, tb = self._to_xy(0, band[0])
        else:
            tt, tb = top, bottom
        c.create_rectangle(tl, tt, tr, tb, fill=_mix(PANEL, OK, 0.2), outline=OK, width=3)

        # --- marker (only once a stroke has arrived)
        if self.marker_force_target is not None or not band:
            mx, my = self._to_xy(self.marker_pos, self.marker_force)
            c.create_oval(mx - 11, my - 11, mx + 11, my + 11,
                          fill=self.marker_colour, outline=TEXT, width=2)

    # -------------------------------------------------------------------- data
    def handle_stroke(self, packet):
        analysis = analyze_stroke(packet)

        peak_position = analysis.get("peak_position_fraction")
        mean_force = analysis.get("mean_force")

        if peak_position is None:
            return

        # ---- stroke force (N) against the baseline mean force ---------------
        stats = self.app.baseline_stats
        force_state = "n/a"         # no baseline -> position only
        if mean_force is not None:
            self.max_force_seen = max(self.max_force_seen, mean_force)
            self.marker_force_target = mean_force
        if stats.ready and mean_force is not None and stats.mean_force:
            ratio = mean_force / stats.mean_force
            if ratio < 1 - FORCE_TOLERANCE:
                force_state = "low"
            elif ratio > 1 + FORCE_TOLERANCE:
                force_state = "high"
            else:
                force_state = "ok"

        # ---- peak position check ----------------------------------------------
        self.marker_target = min(max(peak_position, 0.0), 1.0)
        position_ok = PEAK_POSITION_MIN <= peak_position <= PEAK_POSITION_MAX

        if position_ok and force_state in ("ok", "n/a"):
            msg, colour = "ON TARGET", OK
        else:
            parts = []
            if not position_ok:
                parts.append("PEAK TOO EARLY" if peak_position < PEAK_POSITION_MIN
                             else "PEAK TOO LATE")
            if force_state == "low":
                parts.append("PUSH HARDER")
            elif force_state == "high":
                parts.append("EASE OFF")
            msg, colour = "  |  ".join(parts), BAD

        self.marker_colour = colour
        self.feedback_var.set(msg)
        self.feedback_label.config(fg=colour)


# --------------------------------------------------------------------------
# Condition: force curve
# --------------------------------------------------------------------------
class ForceCurveFrame(TrialFrame):
    def __init__(self, parent, app):
        super().__init__(parent, app)

        self.last_curve = None
        self._build_layout()
        self.start_loops()

    @staticmethod
    def _status_colour(score):
        if score >= 0.75:
            return OK
        if score >= 0.50:
            return WARN
        return BAD

    def _build_layout(self):
        outer = tk.Frame(self, bg=BACKGROUND)
        outer.pack(fill="both", expand=True, padx=35, pady=20)

        self._top_bar(outer, "FORCE CURVE", f"Target {TARGET_SPM:.0f} spm", pady=0, show_spm=True)

        self.feedback_var = tk.StringVar(value="FOLLOW THE BRIGHT ZONE")
        self.feedback_label = tk.Label(outer, textvariable=self.feedback_var, bg=BACKGROUND,
                                       fg=TEXT, font=(self.family, 24, "bold"))
        self.feedback_label.pack(pady=(14, 8))

        # ---- phase indicators (one per drive phase) ----
        phase_row = tk.Frame(outer, bg=BACKGROUND)
        phase_row.pack(fill="x", pady=(0, 8))
        self.phase_labels = {}
        for i, (name, lo, hi) in enumerate(DRIVE_PHASES):
            phase_row.columnconfigure(i, weight=1)
            lbl = tk.Label(phase_row, text=name, bg=PANEL, fg=MUTED,
                           font=(self.family, 12, "bold"),
                           highlightbackground=PANEL_EDGE, highlightthickness=1, pady=6)
            lbl.grid(row=0, column=i, sticky="ew", padx=4)
            self.phase_labels[name] = lbl

        # ---- chart ----
        panel = tk.Frame(outer, bg=PANEL, highlightbackground=PANEL_EDGE, highlightthickness=1)
        panel.pack(fill="both", expand=True, pady=(0, 12))

        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=PANEL)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.97)
        self.ax = self.figure.add_subplot(111)

        self.ax.set_facecolor(PANEL)
        self.ax.set_xlabel("Drive (%)", color=MUTED)
        self.ax.set_ylabel("Force (% of peak)", color=MUTED)
        self.ax.tick_params(colors=MUTED)
        for side in self.ax.spines.values():
            side.set_color(GRID)

        # Gradient: how optimal is each (position, force) point, 1 = on the ideal curve
        y_grid = np.linspace(0, 115, 116)[:, None]
        optimality = np.exp(-0.5 * ((y_grid - IDEAL[None, :]) / SIGMA[None, :]) ** 2)
        self.ax.imshow(optimality, extent=[0, 100, 0, 115], origin="lower",
                       aspect="auto", cmap=OPT_CMAP, vmin=0, vmax=1, zorder=0)

        # phase separators (names are already in the phase row above the chart)
        for name, lo, hi in DRIVE_PHASES:
            if lo > 0:
                self.ax.axvline(lo, color=MUTED, linestyle=":", linewidth=1, alpha=0.6, zorder=1)

        self.ax.plot(XS, IDEAL, color=TEXT, linestyle="--", linewidth=1,
                     alpha=0.55, label="Optimal", zorder=2)
        (self.ghost_line,) = self.ax.plot([], [], color=GHOST, linewidth=2,
                                          label="Previous", zorder=3)
        (self.line,) = self.ax.plot([], [], color=CURVE, linewidth=3,
                                    label="You", zorder=4)

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 115)
        self.ax.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        self.canvas = FigureCanvasTkAgg(self.figure, master=panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

        controls = tk.Frame(outer, bg=BACKGROUND)
        controls.pack(fill="x")

        tk.Button(controls, text="Finish Trial", command=self.app.finish_condition,
                  bg=BAD, fg=BACKGROUND, font=(self.family, 11, "bold"),
                  relief="flat", padx=18, pady=9).pack(side="right")

    def handle_stroke(self,packet):
        # Collect baseline statistics from EVERY stroke of the first baseline
        if self.record_baseline:
            self.app.baseline_stats.add(analyze_stroke(packet))

        power=packet.get("powerWatts")
        pace=packet.get("paceSecondsPer500Meters")
        spm=packet.get("strokesPerMinute")
        timestamp=packet.get("timestampMs")

        self.power_var.set(f"{float(power):.0f}" if power is not None else "–")
        self.pace_var.set(format_pace(pace))
        self.spm_var.set(f"{float(spm):.1f}" if spm is not None else "–")
        self.strokes_var.set(str(self.stroke_count))

        # --------------------------------------------------
        # DISTANCE
        # --------------------------------------------------
        try:
            pace=float(pace)
            timestamp=float(timestamp)/1000.0

            if pace>0:
                speed=500.0/pace

                if self.previous_timestamp is not None and self.previous_speed is not None:
                    dt=timestamp-self.previous_timestamp

                    if dt>0:
                        average_speed=(self.previous_speed+speed)/2
                        self.distance_m+=average_speed*dt

                self.previous_timestamp=timestamp
                self.previous_speed=speed

                self.distance_var.set(f"{self.distance_m:.0f} m")

        except (TypeError,ValueError):
            pass


def main():
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()