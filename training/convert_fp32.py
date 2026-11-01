"""Convert the final FP32 model to TensorFlow Lite and save what the laptop-versus-board comparison needs.

Run from the repository root:
    .venv/bin/python -m training.convert_fp32
Writes to results/fp32/:
    model_fp32.tflite            converted model, no quantization
    model_fp32_data.cc           the same bytes as a C array for the firmware
    test_inputs.npy              the 155 test windows exactly as the model sees them (float32, units of g)
    test_labels.npy              their true classes (0 wing, 1 ring, 2 slope, 3 negative)
    desktop_keras_outputs.npy    Keras probabilities on the laptop
    desktop_tflite_outputs.npy   TensorFlow Lite interpreter probabilities on the laptop
    conversion_check.json        how closely the two laptop outputs agree

Converted with TensorFlow 2.21 because the brief's reference TensorFlow 2.6.0 cannot run on
Apple Silicon (its only macOS build needs AVX instructions); see AI_LOGBOOK.md §1.14.
The batch size is fixed to 1 before converting, as on the board (AI_LOGBOOK.md §1.18).
"""

from __future__ import annotations

import json

import keras
import numpy as np
import tensorflow as tf

from training.dataset import TEST_SPLIT, EvaluationWindows, build_model_inputs, load_recordings
from training.train_fp32 import DATA_DIR, RESULTS_ROOT

RUN_DIR = RESULTS_ROOT / "fp32"
C_ARRAY_NAME = "g_model_fp32"
BYTES_PER_LINE = 12
DEPLOYED_BATCH_SIZE = 1  # the board classifies one window at a time


def main() -> None:
    model = keras.models.load_model(RUN_DIR / "fp32_model.keras")
    model_bytes = convert(with_fixed_batch(model, DEPLOYED_BATCH_SIZE))
    (RUN_DIR / "model_fp32.tflite").write_bytes(model_bytes)
    (RUN_DIR / "model_fp32_data.cc").write_text(as_c_array(model_bytes, C_ARRAY_NAME))
    inputs, labels = build_model_inputs(load_recordings(DATA_DIR, TEST_SPLIT), EvaluationWindows())
    keras_outputs = model.predict(inputs, verbose=0)
    tflite_outputs = run_tflite(model_bytes, inputs)
    save_arrays(inputs=inputs, labels=labels, keras_outputs=keras_outputs, tflite_outputs=tflite_outputs)
    report = {"model_bytes": len(model_bytes), "tensorflow": tf.__version__}
    report |= compare_outputs(keras_outputs, tflite_outputs, labels)
    (RUN_DIR / "conversion_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def with_fixed_batch(model: keras.Model, batch_size: int) -> keras.Sequential:
    """The same layers and weights with a fixed batch size.

    With the batch size left open, the converter turns Flatten into SHAPE, STRIDED_SLICE, PACK
    and RESHAPE to work out the target shape at run time; with it fixed, Flatten is one RESHAPE.
    """
    fixed = keras.Sequential(
        [keras.Input(batch_shape=(batch_size, *model.input_shape[1:]))]
        + [type(layer).from_config(layer.get_config()) for layer in model.layers],
        name=model.name,
    )
    fixed.set_weights(model.get_weights())
    return fixed


def convert(model: keras.Model) -> bytes:
    """Plain FP32 conversion: no optimizations, so every weight stays a 32-bit float."""
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def run_tflite(model_bytes: bytes, inputs: np.ndarray) -> np.ndarray:
    """One example at a time, like the board, through the laptop TensorFlow Lite interpreter."""
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    outputs = []
    for window in inputs:
        interpreter.set_tensor(input_index, window[np.newaxis])
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_index)[0].copy())
    return np.array(outputs)


def save_arrays(**arrays: np.ndarray) -> None:
    names = {
        "inputs": "test_inputs",
        "labels": "test_labels",
        "keras_outputs": "desktop_keras_outputs",
        "tflite_outputs": "desktop_tflite_outputs",
    }
    for key, array in arrays.items():
        np.save(RUN_DIR / f"{names[key]}.npy", array)


def compare_outputs(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict:
    difference = np.abs(reference - candidate)
    return {
        "examples": int(len(labels)),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "same_prediction": int((reference.argmax(axis=1) == candidate.argmax(axis=1)).sum()),
        "reference_accuracy": float((reference.argmax(axis=1) == labels).mean()),
        "candidate_accuracy": float((candidate.argmax(axis=1) == labels).mean()),
    }


def as_c_array(model_bytes: bytes, name: str) -> str:
    """The model as a C source file; TensorFlow Lite Micro needs the buffer 16-byte aligned."""
    lines = [
        ", ".join(f"0x{byte:02x}" for byte in model_bytes[start : start + BYTES_PER_LINE])
        for start in range(0, len(model_bytes), BYTES_PER_LINE)
    ]
    body = ",\n    ".join(lines)
    return (
        f"// Generated by training/convert_fp32.py from results/fp32/fp32_model.keras. Do not edit.\n"
        f"alignas(16) const unsigned char {name}[] = {{\n    {body}\n}};\n"
        f"const unsigned int {name}_len = {len(model_bytes)};\n"
    )


if __name__ == "__main__":
    main()
