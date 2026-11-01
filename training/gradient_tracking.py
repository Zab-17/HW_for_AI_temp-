"""Record how large each layer's gradient is at every training step.

The gradient norm (the length of a layer's gradient vector) shows whether a layer
is still learning (large), has settled (small) or is blowing up (huge). Keras does
not expose gradients during fit(), so this wrapper computes them itself.
"""

from __future__ import annotations

import csv
from pathlib import Path

import keras
import tensorflow as tf

GRADIENT_NORM_PREFIX = "grad_norm_"


class GradientTrackingModel(keras.Model):
    """Trains the wrapped network normally and adds each layer's gradient norm to the batch logs."""

    def __init__(self, network: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def train_step(self, data):
        inputs, labels, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compute_loss(x=inputs, y=labels, y_pred=predictions, sample_weight=sample_weight)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(gradients, self.trainable_variables)
        self._record_loss(loss, batch_size=tf.shape(inputs)[0])
        logs = self.compute_metrics(inputs, labels, predictions, sample_weight=sample_weight)
        logs.update(self._layer_gradient_norms(gradients))
        return logs

    def _record_loss(self, loss, batch_size) -> None:
        """Keras's built-in step updates the running loss itself; a custom step must do it explicitly."""
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss, sample_weight=batch_size)

    def _layer_gradient_norms(self, gradients) -> dict:
        gradient_by_path = {variable.path: g for variable, g in zip(self.trainable_variables, gradients)}
        return {
            f"{GRADIENT_NORM_PREFIX}{layer.name}": tf.linalg.global_norm(
                [gradient_by_path[variable.path] for variable in layer.trainable_variables]
            )
            for layer in self.network.layers
            if layer.trainable_variables
        }


class GradientNormRecorder(keras.callbacks.Callback):
    """Collects the per-layer gradient norms that GradientTrackingModel reports after every batch."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self._epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        norms = {
            name.removeprefix(GRADIENT_NORM_PREFIX): float(value)
            for name, value in logs.items()
            if name.startswith(GRADIENT_NORM_PREFIX)
        }
        self.rows.append({"epoch": self._epoch, "batch": batch + 1, **norms})

    def save_csv(self, path: Path) -> None:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
