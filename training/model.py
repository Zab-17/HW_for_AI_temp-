"""The 32-bit floating-point baseline network fixed by the project brief."""

import keras
from keras import layers

from training.dataset import AXES, CLASS_NAMES, WINDOW_ROWS

DROPOUT_RATE = 0.1


def build_baseline_model() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.Input(shape=(WINDOW_ROWS, AXES, 1), name="accelerometer_window"),
            layers.Conv2D(8, (4, 3), strides=1, padding="same", activation="relu", name="conv1"),
            layers.MaxPooling2D((3, 3), strides=(3, 3), padding="valid", name="pool1"),
            layers.Dropout(DROPOUT_RATE, name="dropout1"),
            layers.Conv2D(16, (4, 1), strides=1, padding="same", activation="relu", name="conv2"),
            layers.MaxPooling2D((3, 1), strides=(3, 1), padding="same", name="pool2"),
            layers.Dropout(DROPOUT_RATE, name="dropout2"),
            layers.Flatten(name="flatten"),
            layers.Dense(16, activation="relu", name="dense1"),
            layers.Dropout(DROPOUT_RATE, name="dropout3"),
            layers.Dense(len(CLASS_NAMES), activation="softmax", name="dense2"),
        ],
        name="magic_wand_fp32",
    )
