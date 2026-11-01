from fractions import Fraction

import numpy as np

from training.augmentation import (
    AUGMENTED_COPIES_PER_RECORDING,
    OFFSET_RANGE_MILLI_G,
    augmented_copies,
    change_speed,
    scale_amplitude,
    shift_offset,
    sliding_windows,
)


def ramp_readings(rows):
    return np.arange(rows * 3, dtype=float).reshape(rows, 3)


def test_change_speed_stretches_length_and_keeps_end_points():
    readings = ramp_readings(60)

    slower = change_speed(readings, Fraction(3, 2))

    assert len(slower) == 90
    np.testing.assert_allclose(slower[[0, -1]], readings[[0, -1]])


def test_scale_amplitude_multiplies_every_reading():
    np.testing.assert_allclose(scale_amplitude(ramp_readings(4), Fraction(3, 2)), ramp_readings(4) * 1.5)


def test_shift_offset_adds_one_constant_within_range():
    readings = ramp_readings(10)

    shifted = shift_offset(readings, np.random.default_rng(0))

    difference = shifted - readings
    assert np.allclose(difference, difference[0, 0])
    assert abs(difference[0, 0]) <= OFFSET_RANGE_MILLI_G


def test_augmented_copies_start_with_the_original_recording():
    readings = ramp_readings(50)

    copies = augmented_copies(readings, np.random.default_rng(0))

    assert len(copies) == AUGMENTED_COPIES_PER_RECORDING
    np.testing.assert_array_equal(copies[0], readings)


def test_sliding_windows_step_through_a_long_recording():
    windows = sliding_windows(ramp_readings(150), window_rows=120, stride=10)

    assert [len(w) for w in windows] == [120, 120, 120, 120]
    np.testing.assert_array_equal(windows[1][0], ramp_readings(150)[10])
