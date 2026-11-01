from pathlib import Path

import numpy as np
import pytest

from training.dataset import (
    TEST_SPLIT,
    TRAINING_SPLIT,
    VALIDATION_SPLIT,
    WINDOW_ROWS,
    EvaluationWindows,
    LabelledRecording,
    NoNoise,
    TrainingWindows,
    UniformJitter,
    build_model_inputs,
    load_recordings,
    pad_after,
    pad_before,
    read_gesture_repetitions,
    read_negative_chunks,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
needs_dataset = pytest.mark.skipif(
    not (DATA_DIR / "wing").is_dir(), reason="Magic Wand dataset not extracted into data/"
)


def ramp_readings(rows):
    return np.arange(rows * 3, dtype=float).reshape(rows, 3)


def write_lines(tmp_path, lines):
    path = tmp_path / "recording.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_gesture_file_is_split_at_markers_and_terminal_noise_is_ignored(tmp_path):
    path = write_lines(
        tmp_path,
        [
            "Script started on Mon Jul 29",
            "bash-3.2$ screen",
            " -,-,-",
            "1.0,2.0,3.0",
            "4.0,5.0,6.0",
            " -,-,-",
            "7.0,8.0,9.0",
        ],
    )

    repetitions = read_gesture_repetitions(path)

    assert [len(r) for r in repetitions] == [2, 1]
    assert repetitions[1].tolist() == [[7.0, 8.0, 9.0]]


def test_negative_recording_is_cut_like_the_original_scripts(tmp_path):
    readings = [f"{i}.0,0.0,0.0" for i in range(1, 251)]

    chunks = read_negative_chunks(write_lines(tmp_path, readings))

    assert [len(c) for c in chunks] == [120, 120, 8]
    assert chunks[1][0, 0] == 122.0, "the reading that triggers a new chunk is skipped"


def test_pad_before_puts_gesture_at_end_and_repeats_first_reading():
    readings = ramp_readings(3)

    window = pad_before(readings, NoNoise())

    assert window.shape == (WINDOW_ROWS, 3)
    np.testing.assert_array_equal(window[-3:], readings)
    np.testing.assert_array_equal(window[0], readings[0])


def test_pad_after_puts_gesture_at_start_and_repeats_last_reading():
    readings = ramp_readings(3)

    window = pad_after(readings, NoNoise())

    np.testing.assert_array_equal(window[:3], readings)
    np.testing.assert_array_equal(window[-1], readings[-1])


def test_long_recording_keeps_its_first_window_rows():
    readings = ramp_readings(WINDOW_ROWS + 47)

    window = pad_after(readings, NoNoise())

    np.testing.assert_array_equal(window, readings[:WINDOW_ROWS])


def test_jitter_only_touches_padding_and_stays_within_amplitude():
    readings = ramp_readings(10)
    jitter = UniformJitter(np.random.default_rng(0), amplitude=10.0)

    window = pad_after(readings, jitter)

    np.testing.assert_array_equal(window[:10], readings)
    assert np.all(np.abs(window[10:] - readings[-1]) <= 10.0)


def test_model_inputs_are_in_units_of_g():
    still_board = LabelledRecording(np.tile([0.0, 0.0, 1000.0], (WINDOW_ROWS, 1)), label=3)

    inputs, _ = build_model_inputs([still_board], EvaluationWindows())

    np.testing.assert_allclose(inputs[0, :, 2, 0], 1.0)


def test_training_windows_give_two_copies_and_evaluation_windows_one():
    readings = ramp_readings(5)
    training = TrainingWindows(UniformJitter(np.random.default_rng(0), amplitude=10.0))

    assert len(training.windows_for(readings)) == 2
    assert len(EvaluationWindows().windows_for(readings)) == 1


@needs_dataset
@pytest.mark.parametrize(
    "split, expected_count", [(TRAINING_SPLIT, 434), (VALIDATION_SPLIT, 92), (TEST_SPLIT, 155)]
)
def test_split_sizes_match_the_project_brief(split, expected_count):
    assert len(load_recordings(DATA_DIR, split)) == expected_count


@needs_dataset
def test_model_inputs_have_the_network_input_shape():
    recordings = load_recordings(DATA_DIR, VALIDATION_SPLIT)

    inputs, labels = build_model_inputs(recordings, EvaluationWindows())

    assert inputs.shape == (92, WINDOW_ROWS, 3, 1)
    assert inputs.dtype == np.float32
    assert set(labels.tolist()) == {0, 1, 2, 3}
