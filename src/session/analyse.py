"""Post-session analysis: when and why was the student not paying attention?

    python -m src.session.analyse                      # newest session
    python -m src.session.analyse sessions/session_X.csv
    python -m src.session.analyse --window 60 --threshold 0.8

Reports overall attention, the periods of sustained inattention (with the
likely cause), attention per time window and per user-marked section, and
where the gaze fell on the screen. A timeline chart is saved next to the CSV.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from .session_logger import SESSIONS_DIR

CAUSES = ("no face", "phone", "looking away", "other")
AWAY_POSES = ("left", "right", "down")


def loadSession(path):
    """Read a session CSV into numpy arrays and lists."""

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path} contains no samples")

    def number(name):
        return np.array(
            [float(r[name]) if r[name] != "" else np.nan for r in rows]
        )

    return {
        "t": number("t"),
        "attention": number("attention"),
        "faces": number("faces"),
        "phone": number("phone"),
        "pose": [r["pose"] for r in rows],
        "gaze_x": number("gaze_x"),
        "gaze_y": number("gaze_y"),
        "off_screen": number("off_screen"),
        "marker": [r["marker"] for r in rows]
    }


def sampleDurations(t, maxGap=1.0):
    """Seconds each row represents; long gaps (lost frames) are capped."""

    if len(t) == 1:
        return np.array([0.0])

    dt = np.diff(t, append=t[-1] + np.median(np.diff(t)))
    return np.clip(dt, 0.0, maxGap)


def causeOf(data, index):
    """Most likely reason a row was classified as inattentive."""

    if data["faces"][index] == 0:
        return "no face"
    if data["phone"][index] == 1:
        return "phone"
    if data["pose"][index] in AWAY_POSES:
        return "looking away"
    return "other"


def summarise(data, durations, mask):
    """Attention and gaze statistics for the rows selected by `mask`."""

    attention = data["attention"]
    weights = durations * mask

    attentive = weights[attention == 0].sum()
    inattentive = weights[attention == 1].sum()
    known = attentive + inattentive

    causes = dict.fromkeys(CAUSES, 0.0)
    for index in np.flatnonzero(mask & (attention == 1)):
        causes[causeOf(data, index)] += durations[index]

    hasGaze = ~np.isnan(data["gaze_x"])
    gazeTime = weights[hasGaze].sum()
    offTime = weights[hasGaze & (data["off_screen"] == 1)].sum()

    return {
        "seconds": float(weights.sum()),
        "attentive": float(attentive),
        "inattentive": float(inattentive),
        "attention": float(attentive / known) if known else None,
        "causes": causes,
        "gaze_seconds": float(gazeTime),
        "off_screen": float(offTime / gazeTime) if gazeTime else None
    }


def findInattentivePeriods(data, durations, minSeconds=3.0, mergeGap=1.0):
    """Runs of inattentive rows, merging short gaps, of at least minSeconds."""

    t = data["t"]
    flagged = np.flatnonzero(data["attention"] == 1)
    runs = []

    for index in flagged:
        end = t[index] + durations[index]

        if runs and t[index] - runs[-1]["end"] <= mergeGap:
            runs[-1]["end"] = end
            runs[-1]["rows"].append(index)
        else:
            runs.append({"start": t[index], "end": end, "rows": [index]})

    periods = []

    for run in runs:
        if run["end"] - run["start"] < minSeconds:
            continue

        counts = dict.fromkeys(CAUSES, 0.0)
        for index in run["rows"]:
            counts[causeOf(data, index)] += durations[index]

        periods.append({
            "start": float(run["start"]),
            "end": float(run["end"]),
            "cause": max(counts, key=counts.get)
        })

    return periods


def splitWindows(duration, window):
    edges = np.arange(0.0, duration, window)
    return [(float(e), float(min(e + window, duration))) for e in edges]


def splitSections(data):
    """Sections delimited by markers; the first starts at the first sample."""

    t = data["t"]
    starts = [(float(t[0]), "start")]

    for index, name in enumerate(data["marker"]):
        if name:
            starts.append((float(t[index]), name))

    end = float(t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0))

    return [
        (name, begin, starts[i + 1][0] if i + 1 < len(starts) else end)
        for i, (begin, name) in enumerate(starts)
    ]


def gazeGrid(data, durations, size=3):
    """Share of gaze time falling in each cell of a size x size screen grid."""

    valid = ~np.isnan(data["gaze_x"])
    total = durations[valid].sum()

    if not total:
        return None

    cells = np.zeros((size, size))
    cols = np.clip((data["gaze_x"][valid] * size).astype(int), 0, size - 1)
    rows = np.clip((data["gaze_y"][valid] * size).astype(int), 0, size - 1)

    for row, col, weight in zip(rows, cols, durations[valid]):
        cells[row, col] += weight

    return cells / total


def analyse(data, window=30.0, threshold=0.7, minPeriod=3.0):
    """Run every analysis; returns plain dicts and lists for reporting."""

    t = data["t"]
    durations = sampleDurations(t)
    everything = np.ones(len(t), dtype=bool)

    def inRange(start, end):
        return (t >= start) & (t < end)

    windows = [
        {"start": start, "end": end,
         **summarise(data, durations, inRange(start, end))}
        for start, end in splitWindows(float(t[-1] + durations[-1]), window)
    ]

    sections = [
        {"name": name, "start": start, "end": end,
         **summarise(data, durations, inRange(start, end))}
        for name, start, end in splitSections(data)
    ]

    return {
        "overall": summarise(data, durations, everything),
        "periods": findInattentivePeriods(data, durations, minPeriod),
        "windows": windows,
        "sections": sections,
        "grid": gazeGrid(data, durations),
        "threshold": threshold,
        "window": window
    }


def _clock(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _percent(value):
    return "  n/a" if value is None else f"{value * 100:4.0f}%"


def _topCause(summary):
    causes = summary["causes"]
    best = max(causes, key=causes.get)
    return best if causes[best] > 0 else "-"


def formatReport(result):
    overall = result["overall"]
    threshold = result["threshold"]
    lines = ["SESSION REPORT", "=" * 62]

    lines.append(
        f"Duration {_clock(overall['seconds'])}   "
        f"attentive {_percent(overall['attention']).strip()}   "
        f"(attentive {_clock(overall['attentive'])}, "
        f"inattentive {_clock(overall['inattentive'])})"
    )

    if overall["inattentive"]:
        parts = [
            f"{name} {share / overall['inattentive'] * 100:.0f}%"
            for name, share in overall["causes"].items() if share
        ]
        lines.append("Inattention was due to: " + ", ".join(parts))

    if overall["off_screen"] is not None:
        lines.append(
            f"Gaze off-screen {_percent(overall['off_screen']).strip()} "
            f"of the {_clock(overall['gaze_seconds'])} with gaze data"
        )

    lines += ["", "Sustained inattention periods", "-" * 62]

    if result["periods"]:
        for period in result["periods"]:
            lines.append(
                f"{_clock(period['start'])} - {_clock(period['end'])}  "
                f"({period['end'] - period['start']:.0f} s)  "
                f"{period['cause']}"
            )
    else:
        lines.append("none")

    def table(title, rows, label):
        lines.extend(["", title, "-" * 62,
                      f"{label:<14}{'time':<14}{'attention':>10}  "
                      f"{'off-screen':>10}  main cause"])
        for row in rows:
            flag = (
                "  <-- LOW"
                if row["attention"] is not None
                and row["attention"] < threshold else ""
            )
            name = row.get("name") or _clock(row["start"])
            lines.append(
                f"{name:<14}"
                f"{_clock(row['start'])}-{_clock(row['end']):<8}"
                f"{_percent(row['attention']):>10}  "
                f"{_percent(row['off_screen']):>10}  "
                f"{_topCause(row)}{flag}"
            )

    if len(result["sections"]) > 1:
        table("Attention per marked section", result["sections"], "section")

    table(f"Attention per {result['window']:.0f} s window",
          result["windows"], "window")

    lowest = sorted(
        (w for w in result["windows"] if w["attention"] is not None),
        key=lambda w: w["attention"]
    )[:3]
    if lowest:
        lines += ["", "Least attentive windows: " + ", ".join(
            f"{_clock(w['start'])}-{_clock(w['end'])} "
            f"({_percent(w['attention']).strip()})" for w in lowest
        )]

    grid = result["grid"]
    if grid is not None:
        lines += ["", "Gaze distribution over the screen (rows top to bottom)",
                  "-" * 62]
        for row in grid:
            lines.append("   ".join(f"{v * 100:4.0f}%" for v in row))

    return "\n".join(lines)


def saveTimeline(data, result, path):
    """Plot attention, gaze and markers against time; returns True on success."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    t = data["t"]
    durations = sampleDurations(t)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]}
    )

    for value, colour in ((0, "#3cb371"), (1, "#d9534f")):
        spans = [(t[i], durations[i]) for i in np.flatnonzero(data["attention"] == value)]
        top.broken_barh(spans, (0, 1), facecolors=colour, linewidth=0)

    top.set_yticks([])
    top.set_title("Attention (green = attentive, red = inattentive)")

    hasGaze = ~np.isnan(data["gaze_x"])
    if hasGaze.any():
        bottom.plot(t[hasGaze], data["gaze_x"][hasGaze], ".", ms=2, label="gaze x")
        bottom.plot(t[hasGaze], data["gaze_y"][hasGaze], ".", ms=2, label="gaze y")
        bottom.legend(loc="upper right")
    bottom.set_ylim(1.05, -0.05)
    bottom.set_ylabel("screen position")
    bottom.set_xlabel("time (s)")

    for axis in (top, bottom):
        for name, begin, _ in splitSections(data)[1:]:
            axis.axvline(begin, color="black", linestyle="--", linewidth=1)

    for window in result["windows"]:
        if window["attention"] is not None and window["attention"] < result["threshold"]:
            top.axvspan(window["start"], window["end"], color="red", alpha=0.08)

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return True


def latestSession(directory=SESSIONS_DIR):
    files = sorted(Path(directory).glob("session_*.csv"))

    if not files:
        raise FileNotFoundError(f"No session CSV files found in {directory}")

    return files[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse a recorded session."
    )
    parser.add_argument("csv", nargs="?", type=Path, default=None,
                        help="Session CSV (default: newest in sessions/)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="Window length in seconds (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Attention below this is flagged LOW (default: 0.7)")
    parser.add_argument("--min-period", type=float, default=3.0,
                        help="Shortest inattention period to list, in seconds")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    path = args.csv or latestSession()
    data = loadSession(path)
    result = analyse(data, args.window, args.threshold, args.min_period)
    report = formatReport(result)

    print(f"Session: {path}\n")
    print(report)

    path = Path(path)
    path.with_name(path.stem + "_report.txt").write_text(
        report, encoding="utf-8"
    )

    if not args.no_plot:
        chart = path.with_name(path.stem + "_timeline.png")
        if saveTimeline(data, result, chart):
            print(f"\nTimeline chart: {chart}")


if __name__ == "__main__":
    main()
