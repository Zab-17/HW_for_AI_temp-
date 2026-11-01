"""Training-only data augmentation (allowed by the course FAQ; validation and test stay unchanged).

The speed, size and offset variants copy the original TensorFlow Magic Wand script
(train/data_augmentation.py at the pinned commit): seven time-warp fractions, the
same seven fractions as amplitude scales, and five random constant offsets.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

ORIGINAL_FRACTIONS = (
    Fraction(3, 2),
    Fraction(5, 3),
    Fraction(2, 3),
    Fraction(3, 4),
    Fraction(9, 5),
    Fraction(6, 5),
    Fraction(4, 5),
)
# The original warps a recording of n rows to about n * denominator / molecule rows.
SPEED_FACTORS = tuple(1 / fraction for fraction in ORIGINAL_FRACTIONS)
AMPLITUDE_FACTORS = ORIGINAL_FRACTIONS
OFFSET_COPIES = 5
OFFSET_RANGE_MILLI_G = 100.0  # the original adds (random - 0.5) * 200
AUGMENTED_COPIES_PER_RECORDING = 1 + len(SPEED_FACTORS) + len(AMPLITUDE_FACTORS) + OFFSET_COPIES


def augmented_copies(readings: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """The recording itself followed by its speed, size and offset variants."""
    return [
        readings,
        *(change_speed(readings, factor) for factor in SPEED_FACTORS),
        *(scale_amplitude(readings, factor) for factor in AMPLITUDE_FACTORS),
        *(shift_offset(readings, rng) for _ in range(OFFSET_COPIES)),
    ]


def change_speed(readings: np.ndarray, length_factor: Fraction) -> np.ndarray:
    """Resample to length_factor times as many rows: above 1 the gesture is drawn slower, below 1 faster."""
    new_rows = max(2, round(len(readings) * length_factor))
    old_time = np.linspace(0.0, 1.0, len(readings))
    new_time = np.linspace(0.0, 1.0, new_rows)
    return np.column_stack([np.interp(new_time, old_time, axis) for axis in readings.T])


def scale_amplitude(readings: np.ndarray, factor: Fraction) -> np.ndarray:
    return readings * float(factor)


def shift_offset(readings: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return readings + rng.uniform(-OFFSET_RANGE_MILLI_G, OFFSET_RANGE_MILLI_G)


def sliding_windows(readings: np.ndarray, window_rows: int, stride: int) -> list[np.ndarray]:
    """Overlapping slices of a long recording, used to get many more real no-gesture examples."""
    starts = range(0, len(readings) - window_rows + 1, stride)
    return [readings[start : start + window_rows] for start in starts]
