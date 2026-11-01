import keras
import numpy as np

from training.gradient_tracking import GradientNormRecorder, GradientTrackingModel
from training.model import build_baseline_model


def train_one_epoch_on_random_data():
    keras.utils.set_random_seed(0)
    trainer = GradientTrackingModel(build_baseline_model())
    trainer.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    recorder = GradientNormRecorder()
    inputs = np.random.default_rng(0).normal(size=(64, 128, 3, 1)).astype(np.float32)
    labels = np.arange(64, dtype=np.int32) % 4
    history = trainer.fit(inputs, labels, epochs=1, batch_size=32, callbacks=[recorder], verbose=0)
    return history, recorder


def test_training_loss_is_recorded():
    history, _ = train_one_epoch_on_random_data()

    assert history.history["loss"][0] > 0


def test_every_batch_records_a_gradient_norm_for_each_trainable_layer():
    _, recorder = train_one_epoch_on_random_data()

    assert len(recorder.rows) == 2
    assert set(recorder.rows[0]) == {"epoch", "batch", "conv1", "conv2", "dense1", "dense2"}
