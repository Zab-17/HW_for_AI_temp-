"""Train the 32-bit floating-point baseline (Requirement 1) and save everything needed to analyse it.

Run from the repository root:
    .venv/bin/python -m training.train_fp32
Outputs go to results/<run-name>/ (default results/fp32/).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from training.augmentation import augmented_copies, sliding_windows
from training.dataset import (
    CLASS_NAMES,
    NEGATIVE,
    NEGATIVE_CHUNK_ROWS,
    TEST_SPLIT,
    TRAINING_SPLIT,
    VALIDATION_SPLIT,
    EvaluationWindows,
    LabelledRecording,
    TrainingWindows,
    UniformJitter,
    build_model_inputs,
    load_recordings,
    negative_recording_paths,
    read_all_readings,
)
from training.gradient_tracking import GRADIENT_NORM_PREFIX, GradientNormRecorder, GradientTrackingModel
from training.model import build_baseline_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
RESULTS_ROOT = REPOSITORY_ROOT / "results"
TEST_ACCURACY_TARGET = 0.90


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 15
    jitter_amplitude: float = 10.0  # thousandths of g, same as the original scripts
    random_seed: int = 42
    augment: bool = True  # original speed/size/offset variants of every training recording
    negative_window_stride: int = 10  # rows between overlapping no-gesture windows cut from negative_3
    run_name: str = "fp32"  # results are saved in results/<run_name>/


@dataclass(frozen=True)
class PreparedSplits:
    training: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]

    def by_name(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {"training": self.training, "validation": self.validation, "test": self.test}


def main() -> None:
    settings = parse_settings()
    output_dir = RESULTS_ROOT / settings.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_training(settings, DATA_DIR, output_dir)


def parse_settings() -> TrainingSettings:
    """Every TrainingSettings field can be overridden from the command line, e.g. --batch-size 64."""
    parser = argparse.ArgumentParser(description="Train the FP32 baseline.")
    for setting in fields(TrainingSettings):
        flag = f"--{setting.name.replace('_', '-')}"
        if isinstance(setting.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=setting.default)
        else:
            parser.add_argument(flag, type=type(setting.default), default=setting.default)
    return TrainingSettings(**vars(parser.parse_args()))


def run_training(settings: TrainingSettings, data_dir: Path, output_dir: Path) -> None:
    keras.utils.set_random_seed(settings.random_seed)
    tf.config.experimental.enable_op_determinism()
    splits = prepare_splits(settings, data_dir)
    trainer = compile_trainer(settings)
    recorder = GradientNormRecorder()
    history = trainer.fit(
        *splits.training,
        validation_data=splits.validation,
        epochs=settings.max_epochs,
        batch_size=settings.batch_size,
        class_weight=balanced_class_weights(splits.training[1]),
        callbacks=[early_stopping(settings), recorder],
        verbose=2,
    )
    save_results(output_dir, settings, trainer, history, recorder, splits)


def prepare_splits(settings: TrainingSettings, data_dir: Path) -> PreparedSplits:
    augmentation_rng, jitter_rng = (
        np.random.default_rng(seed) for seed in np.random.SeedSequence(settings.random_seed).spawn(2)
    )
    jitter = UniformJitter(jitter_rng, settings.jitter_amplitude)
    return PreparedSplits(
        training=build_model_inputs(
            training_recordings(settings, data_dir, augmentation_rng), TrainingWindows(jitter)
        ),
        validation=build_model_inputs(load_recordings(data_dir, VALIDATION_SPLIT), EvaluationWindows()),
        test=build_model_inputs(load_recordings(data_dir, TEST_SPLIT), EvaluationWindows()),
    )


def training_recordings(
    settings: TrainingSettings, data_dir: Path, rng: np.random.Generator
) -> list[LabelledRecording]:
    """The 434 brief recordings; when augmenting, plus extra no-gesture windows and all variants."""
    recordings = load_recordings(data_dir, TRAINING_SPLIT)
    if not settings.augment:
        return recordings
    recordings += extra_negative_windows(data_dir, settings.negative_window_stride)
    return [
        LabelledRecording(copy, recording.label)
        for recording in recordings
        for copy in augmented_copies(recording.readings, rng)
    ]


def extra_negative_windows(data_dir: Path, stride: int) -> list[LabelledRecording]:
    return [
        LabelledRecording(window, CLASS_NAMES.index(NEGATIVE))
        for path in negative_recording_paths(data_dir, TRAINING_SPLIT)
        for window in sliding_windows(read_all_readings(path), NEGATIVE_CHUNK_ROWS, stride)
    ]


def compile_trainer(settings: TrainingSettings) -> GradientTrackingModel:
    trainer = GradientTrackingModel(build_baseline_model())
    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=settings.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return trainer


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Each class contributes equally to the loss, however many examples it has."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {label: len(labels) / (len(CLASS_NAMES) * count) for label, count in enumerate(counts)}


def early_stopping(settings: TrainingSettings) -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=settings.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )


def save_results(output_dir, settings, trainer, history, recorder, splits) -> None:
    trainer.network.save(output_dir / "fp32_model.keras")
    recorder.save_csv(output_dir / "gradient_norms.csv")
    save_history_csv(history, output_dir / "training_history.csv")
    summary = summarise_run(settings, trainer, history, splits)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["accuracy"], indent=2))


def save_history_csv(history: keras.callbacks.History, path: Path) -> None:
    columns = [name for name in history.history if not name.startswith(GRADIENT_NORM_PREFIX)]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *columns])
        for epoch_index in range(len(history.epoch)):
            writer.writerow([epoch_index + 1, *(history.history[name][epoch_index] for name in columns)])


def summarise_run(settings, trainer, history, splits: PreparedSplits) -> dict:
    named_splits = splits.by_name()
    scores = {
        name: trainer.evaluate(*split, verbose=0, return_dict=True) for name, split in named_splits.items()
    }
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    test_accuracy = scores["test"]["accuracy"]
    return {
        "settings": asdict(settings),
        "windows_after_padding_and_augmentation": {
            name: int(len(labels)) for name, (_, labels) in named_splits.items()
        },
        "class_weights": {
            CLASS_NAMES[k]: round(v, 3) for k, v in balanced_class_weights(splits.training[1]).items()
        },
        "epochs_run": len(history.epoch),
        "best_epoch": best_epoch,
        "accuracy": {name: round(score["accuracy"], 4) for name, score in scores.items()},
        "loss": {name: round(score["loss"], 4) for name, score in scores.items()},
        "meets_90_percent_test_target": bool(test_accuracy >= TEST_ACCURACY_TARGET),
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__},
    }


if __name__ == "__main__":
    main()
