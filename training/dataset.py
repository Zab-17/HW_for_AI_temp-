"""Turn the Magic Wand accelerometer recordings into fixed-size model inputs.

The split comes from the project brief. Chunking and padding follow the original
TensorFlow Magic Wand scripts (tensorflow/lite/micro/examples/magic_wand/train,
data_prepare.py and data_load.py at the pinned commit), so the example counts
match the brief: 434 training, 92 validation, 155 test recordings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

WINDOW_ROWS = 128
AXES = 3
GESTURES = ("wing", "ring", "slope")
NEGATIVE = "negative"
CLASS_NAMES = GESTURES + (NEGATIVE,)  # position = class number, same order as the firmware
NEGATIVE_CHUNK_ROWS = 120
MILLI_G_PER_G = 1000.0
REPETITION_MARKER = "-,-,-"
READING_LINE = re.compile(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?$")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    people: tuple[str, ...]
    negative_recordings: tuple[int, ...]


TRAINING_SPLIT = SplitSpec("training", ("hyw", "shiyun", "tangsy", "dengyl", "jiangyh", "xunkai"), (3,))
VALIDATION_SPLIT = SplitSpec("validation", ("pengxl",), (2, 4))
TEST_SPLIT = SplitSpec("test", ("liucx", "zhangxy"), (1, 5))


@dataclass(frozen=True)
class LabelledRecording:
    readings: np.ndarray  # shape (rows, 3), x/y/z in thousandths of g
    label: int


def load_recordings(data_dir: Path, split: SplitSpec) -> list[LabelledRecording]:
    return _gesture_recordings(data_dir, split) + _negative_recordings(data_dir, split)


def _gesture_recordings(data_dir: Path, split: SplitSpec) -> list[LabelledRecording]:
    return [
        LabelledRecording(repetition, CLASS_NAMES.index(gesture))
        for gesture in GESTURES
        for person in split.people
        for repetition in read_gesture_repetitions(data_dir / gesture / f"output_{gesture}_{person}.txt")
    ]


def _negative_recordings(data_dir: Path, split: SplitSpec) -> list[LabelledRecording]:
    return [
        LabelledRecording(chunk, CLASS_NAMES.index(NEGATIVE))
        for path in negative_recording_paths(data_dir, split)
        for chunk in read_negative_chunks(path)
    ]


def negative_recording_paths(data_dir: Path, split: SplitSpec) -> list[Path]:
    return [data_dir / NEGATIVE / f"output_negative_{number}.txt" for number in split.negative_recordings]


def read_gesture_repetitions(path: Path) -> list[np.ndarray]:
    """One array per repetition; every repetition starts after a '-,-,-' line."""
    repetitions: list[list[list[float]]] = []
    for line in _stripped_lines(path):
        if line.endswith(REPETITION_MARKER):
            repetitions.append([])
        elif repetitions and READING_LINE.match(line):
            repetitions[-1].append(_parse_reading(line))
    return [np.array(rows) for rows in repetitions if rows]


def read_negative_chunks(path: Path) -> list[np.ndarray]:
    """Cut one long no-gesture recording into chunks exactly as the original scripts did.

    The original keeps 120 readings per chunk, skips the reading that triggers the
    next chunk, and keeps the shorter leftover chunk at the end. Reproducing this
    quirk is what makes the counts match the brief.
    """
    chunks: list[list[list[float]]] = [[]]
    for reading in read_all_readings(path).tolist():
        if len(chunks[-1]) == NEGATIVE_CHUNK_ROWS:
            chunks.append([])
        else:
            chunks[-1].append(reading)
    return [np.array(rows) for rows in chunks if rows]


def read_all_readings(path: Path) -> np.ndarray:
    """Every reading in the file as one (rows, 3) array, ignoring markers and terminal noise."""
    return np.array([_parse_reading(line) for line in _stripped_lines(path) if READING_LINE.match(line)])


def _stripped_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(errors="replace").splitlines()]


def _parse_reading(line: str) -> list[float]:
    return [float(value) for value in line.split(",")]


class Noise(Protocol):
    def sample(self, rows: int) -> np.ndarray: ...


class UniformJitter:
    """Uniform noise in [-amplitude, +amplitude): a hand holding the board still is never perfectly still."""

    def __init__(self, rng: np.random.Generator, amplitude: float):
        self._rng = rng
        self._amplitude = amplitude

    def sample(self, rows: int) -> np.ndarray:
        return self._rng.uniform(-self._amplitude, self._amplitude, size=(rows, AXES))


class NoNoise:
    def sample(self, rows: int) -> np.ndarray:
        return np.zeros((rows, AXES))


def pad_before(readings: np.ndarray, noise: Noise) -> np.ndarray:
    """Gesture at the end of the window; the start repeats the first reading (board held still before)."""
    kept = truncate_to_window(readings)
    return np.concatenate([_edge_fill(kept[0], WINDOW_ROWS - len(kept), noise), kept])


def pad_after(readings: np.ndarray, noise: Noise) -> np.ndarray:
    """Gesture at the start of the window; the end repeats the last reading (board held still after)."""
    kept = truncate_to_window(readings)
    return np.concatenate([kept, _edge_fill(kept[-1], WINDOW_ROWS - len(kept), noise)])


def truncate_to_window(readings: np.ndarray) -> np.ndarray:
    """Over-long recordings keep their first rows, as in the original scripts (affects 4 of 681)."""
    return readings[:WINDOW_ROWS]


def _edge_fill(edge_reading: np.ndarray, rows: int, noise: Noise) -> np.ndarray:
    return edge_reading + noise.sample(rows)


class WindowPolicy(Protocol):
    def windows_for(self, readings: np.ndarray) -> list[np.ndarray]: ...


class TrainingWindows:
    """Two jittered copies per recording, gesture at the end and at the start (original recipe)."""

    def __init__(self, jitter: Noise):
        self._jitter = jitter

    def windows_for(self, readings: np.ndarray) -> list[np.ndarray]:
        return [pad_before(readings, self._jitter), pad_after(readings, self._jitter)]


class EvaluationWindows:
    """One deterministic copy, gesture at the end: what the board sees right after a gesture."""

    def windows_for(self, readings: np.ndarray) -> list[np.ndarray]:
        return [pad_before(readings, NoNoise())]


def build_model_inputs(
    recordings: list[LabelledRecording], policy: WindowPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Returns inputs in units of g, shaped (examples, 128, 3, 1) as float32, and integer labels."""
    windows, labels = [], []
    for recording in recordings:
        for window in policy.windows_for(recording.readings):
            windows.append(window)
            labels.append(recording.label)
    inputs = to_g(np.stack(windows)).astype(np.float32)[..., np.newaxis]
    return inputs, np.array(labels, dtype=np.int32)


def to_g(readings_in_milli_g: np.ndarray) -> np.ndarray:
    """Raw readings reach ±2,000; at that scale the first run's ReLU neurons died (49.7% test accuracy).

    The firmware must apply the same division before every inference.
    """
    return readings_in_milli_g / MILLI_G_PER_G
