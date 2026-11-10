"""Gaze calibration: maps eye/head features to normalised screen coordinates.

Coordinates are normalised to [0, 1] (0, 0 = top-left of the screen), so the
result does not depend on the display resolution and can later be mapped to
page regions directly.
"""

import ctypes
import time
from pathlib import Path

import cv2
import numpy as np


class CalibrationAborted(Exception):
    """Raised when the user cancels calibration or validation."""


class GazeMapper:
    """Ridge regression from gaze feature vectors to screen coordinates."""

    def __init__(self, degree=2, ridge=1.0):
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")

        self.degree = degree
        self.ridge = ridge
        self.mean = None
        self.std = None
        self.weights = None
        self.screenSize = None

    @property
    def isFitted(self):
        return self.weights is not None

    def _design(self, features):
        z = (features - self.mean) / self.std

        columns = [np.ones(len(z)), *z.T]

        if self.degree == 2:
            # Quadratic terms in the two eye-in-head features (h, v)
            columns += [z[:, 0] * z[:, 0], z[:, 1] * z[:, 1], z[:, 0] * z[:, 1]]

        return np.column_stack(columns)

    def fit(self, features, targets, screenSize=None):
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("features and targets must have the same length")

        self.mean = features.mean(axis=0)
        std = features.std(axis=0)
        # A constant feature carries no information; avoid dividing by zero
        self.std = np.where(std < 1e-9, 1.0, std)

        design = self._design(features)

        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0  # never shrink the bias term

        self.weights = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets
        )

        if screenSize is not None:
            self.screenSize = tuple(int(v) for v in screenSize)

        return self

    def predict(self, features):
        """Predict normalised screen coordinates for one or many vectors."""

        if not self.isFitted:
            raise RuntimeError("GazeMapper has not been fitted")

        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        result = self._design(features) @ self.weights

        return result[0] if len(result) == 1 else result

    def pixelError(self, features, targets, screenSize):
        """Mean distance in pixels between predictions and targets."""

        predicted = np.atleast_2d(self.predict(features))
        difference = (predicted - np.asarray(targets)) * np.asarray(screenSize)
        return float(np.linalg.norm(difference, axis=1).mean())

    def leavePointOut(self, features, targets, groups, screenSize):
        """Cross-validate by holding out one calibration point at a time.

        Returns {group: mean pixel error}. Corner points have to be
        extrapolated, so this is a pessimistic estimate of accuracy.
        """

        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        groups = np.asarray(groups)

        errors = {}

        for group in np.unique(groups):
            held = groups == group

            model = GazeMapper(self.degree, self.ridge).fit(
                features[~held],
                targets[~held]
            )
            errors[int(group)] = model.pixelError(
                features[held],
                targets[held],
                screenSize
            )

        return errors

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            degree=self.degree,
            ridge=self.ridge,
            mean=self.mean,
            std=self.std,
            weights=self.weights,
            screenSize=np.array(self.screenSize or (0, 0))
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)

        mapper = cls(int(data["degree"]), float(data["ridge"]))
        mapper.mean = data["mean"]
        mapper.std = data["std"]
        mapper.weights = data["weights"]

        screenSize = tuple(int(v) for v in data["screenSize"])
        mapper.screenSize = screenSize if all(screenSize) else None

        return mapper


class GazeSmoother:
    """Exponential moving average to reduce jitter in the gaze point."""

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None

    def update(self, point):
        point = np.asarray(point, dtype=np.float64)

        if self.value is None:
            self.value = point
        else:
            self.value = self.alpha * point + (1 - self.alpha) * self.value

        return self.value

    def reset(self):
        self.value = None


def getScreenSize(default=(1920, 1080)):
    """Primary display size in pixels (Windows), or the default."""

    try:
        user32 = ctypes.windll.user32
        size = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        if all(size):
            return size
    except (AttributeError, OSError):
        pass

    return default


def gridPoints(count, margin=0.1):
    """count x count targets in normalised coordinates, as (x, y) rows."""

    if count < 2:
        raise ValueError("grid must be at least 2")

    ticks = np.linspace(margin, 1 - margin, count)
    return [(float(x), float(y)) for y in ticks for x in ticks]


def openFullscreen(windowName):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        windowName,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


def _drawTarget(canvas, point, progress, message):
    height, width = canvas.shape[:2]
    center = (int(point[0] * width), int(point[1] * height))

    # Ring shrinks onto the dot while the user settles, then fills as
    # samples are collected
    ring = int(40 - 28 * min(progress, 1.0))
    cv2.circle(canvas, center, ring, (255, 255, 255), 2)
    cv2.circle(canvas, center, 8, (0, 0, 255), -1)

    cv2.putText(
        canvas,
        message,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )


def collectSamples(
    tracker,
    capture,
    points,
    screenSize,
    windowName,
    settleSeconds=0.8,
    samplesPerPoint=25,
    timeoutSeconds=15.0
):
    """Show each target and record gaze features while the user looks at it.

    Returns (features, targets, groups), one row per usable frame.
    """

    width, height = screenSize
    features, targets, groups = [], [], []

    for index, point in enumerate(points):
        collected = 0
        start = time.monotonic()

        while collected < samplesPerPoint:
            elapsed = time.monotonic() - start

            if elapsed > timeoutSeconds:
                raise RuntimeError(
                    f"No usable face/eye samples for point {index + 1}; "
                    "check lighting and that your face is visible"
                )

            ret, frame = capture.read()

            if not ret:
                raise RuntimeError("Camera stopped providing frames")

            sample = tracker.process(frame)
            settling = elapsed < settleSeconds

            if sample is not None and not sample.blink and not settling:
                features.append(sample.vector())
                targets.append(point)
                groups.append(index)
                collected += 1

            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if settling:
                progress = elapsed / max(settleSeconds, 1e-6)
            else:
                progress = 1.0

            message = (
                f"Look at the dot  ({index + 1}/{len(points)})"
                if sample is not None
                else "No face detected - face the camera"
            )

            _drawTarget(canvas, point, progress, message)
            cv2.imshow(windowName, canvas)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                raise CalibrationAborted()

    return (
        np.array(features, dtype=np.float64),
        np.array(targets, dtype=np.float64),
        np.array(groups, dtype=int)
    )


def calibrate(
    tracker,
    capture,
    screenSize,
    windowName,
    grid=4,
    degree=2,
    ridge=1.0,
    settleSeconds=0.8,
    samplesPerPoint=25
):
    """Run the calibration routine and return a fitted GazeMapper."""

    points = gridPoints(grid)

    features, targets, groups = collectSamples(
        tracker,
        capture,
        points,
        screenSize,
        windowName,
        settleSeconds=settleSeconds,
        samplesPerPoint=samplesPerPoint
    )

    mapper = GazeMapper(degree=degree, ridge=ridge).fit(
        features,
        targets,
        screenSize
    )

    trainError = mapper.pixelError(features, targets, screenSize)
    heldOut = mapper.leavePointOut(features, targets, groups, screenSize)

    print(f"Calibration samples: {len(features)} over {len(points)} points")
    print(f"Training error:          {trainError:.0f} px")
    print(
        f"Leave-one-point-out error: mean {np.mean(list(heldOut.values())):.0f} px, "
        f"worst {max(heldOut.values()):.0f} px (pessimistic: edges extrapolate)"
    )

    return mapper


def validate(
    tracker,
    capture,
    mapper,
    screenSize,
    windowName,
    settleSeconds=0.8,
    samplesPerPoint=25
):
    """Measure accuracy at fresh targets that were not used in calibration."""

    points = [
        (0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)
    ]

    features, targets, groups = collectSamples(
        tracker,
        capture,
        points,
        screenSize,
        windowName,
        settleSeconds=settleSeconds,
        samplesPerPoint=samplesPerPoint
    )

    perPoint = {}

    for index in range(len(points)):
        held = groups == index
        perPoint[index] = mapper.pixelError(
            features[held],
            targets[held],
            screenSize
        )

    mean = float(np.mean(list(perPoint.values())))

    print(f"Validation error: mean {mean:.0f} px")
    for index, error in perPoint.items():
        print(f"  point {index + 1} {points[index]}: {error:.0f} px")

    return mean
