# Report notes — Requirement 1: FP32 baseline deployment and functional validation

Facts and numbers for the report. Sources: `AI_LOGBOOK.md` §1.10–1.18 and the files in `results/`.

## 1. Training (part A, done)
- **Data:** Magic Wand dataset with the brief's person split: 434 training, 92 validation and 155 test recordings. The counts are reproduced exactly using the FAQ's chunking rule for negative recordings.
- **Padding to 128 rows:**
  - Training: two copies per recording, gesture at the end and at the start. The padding repeats the edge reading plus ±10 milli-g uniform jitter (original TensorFlow Magic Wand recipe).
  - Validation and test: one deterministic copy, gesture at the end, padding repeats the first reading. Recordings over 128 rows keep their first 128 rows (4 of 681).
- **Input scaling:** readings in milli-g are divided by 1000, so the inputs are in g. The firmware must apply the same division.
- **Augmentation (training only, real recordings only):**
  - 7 speed variants, 7 amplitude variants and 5 offset variants per recording (original recipe).
  - Overlapping 120-row windows every 10 rows of negative_3.
  - Total: 25,800 training windows.
- **Settings:** Adam, learning rate 0.001, batch size 32, up to 100 epochs, early stopping on validation accuracy (patience 15, best weights restored), balanced class weights, seed 42.
- **Runs** (each change chosen on validation results):

| Run | Change | Epochs (best) | Validation | Test |
|---|---|---|---|---|
| 1 | raw milli-g inputs | 17 (2) | 54.3% | 49.7% |
| 2 | inputs ÷ 1000 | 23 (8) | 79.3% | 75.5% |
| 3 (final) | + augmentation from real recordings | 35 (20) | 81.5% | **91.6%** |

  - **Why run 1 failed:** inputs up to ±2,000 caused gradient norms up to about 1,860, and the first dense layer's ReLU units died. 27 of 476 batches had exactly zero gradient in both convolutions and the first dense layer (`results/fp32_run1_raw_inputs/gradient_norms.png`).
  - **What run 2 showed:** the "negative" class was the weak one (16/33 correct on validation), because training has only 19 negative examples.
  - **A rejected experiment** (1,000 computer-generated still-board negatives) is documented in AI_LOGBOOK §1.13 and removed from the pipeline.
- **Known limitations of the final model:**
  - Seeds 43 and 44 give 89.0% and 89.7% test accuracy.
  - Validation negatives are 17/33 correct.
  - A board lying still is classified as a gesture, because every training gesture ends in flat padding while no negative example does.

## 2. Conversion and the deployed graph (part B, laptop side done)
- **Converter:** TensorFlow 2.21, no quantization, batch size fixed to 1.
- **Model file:** `results/fp32/model_fp32.tflite`, 20,772 bytes.
- **Deviation from the brief:** the reference environment is TensorFlow 2.6.0 (commit 919f693). TensorFlow 2.6.0 cannot run on Apple Silicon (its macOS build requires AVX instructions). Every operator in the converted file is registered in the pinned TensorFlow Lite Micro `all_ops_resolver.cc`, and the pinned resolver ignores operator versions.
- **Operators that execute on the board, and how they relate to the Keras model:**

| Keras layer | Operator on the board | What happened |
|---|---|---|
| conv1 (Conv2D + ReLU) | CONV_2D, fused activation RELU | **fused**: ReLU applied while each output is written, no separate pass over memory |
| pool1 | MAX_POOL_2D | unchanged |
| dropout1, dropout2, dropout3 | none | **removed**: Dropout is the identity at inference |
| conv2 (Conv2D + ReLU) | CONV_2D, RELU | **fused** |
| pool2 | MAX_POOL_2D | unchanged |
| flatten | RESHAPE | **represented differently**: a reshape to [1, 224]; no data moves |
| dense1 (Dense + ReLU) | FULLY_CONNECTED, RELU | **fused**; weights stored transposed as [16, 224] |
| dense2 (Dense, softmax activation) | FULLY_CONNECTED + SOFTMAX | **represented differently**: softmax becomes its own operator, because it needs all 4 outputs (it divides by their sum), whereas ReLU works on one value at a time |

- **Finding about Flatten:** with the batch size left flexible (the converter's default), Flatten became four operators: SHAPE → STRIDED_SLICE → PACK → RESHAPE. Those compute the target shape [batch, 224] on every inference, and the board always has batch 1. Fixing the batch size to 1 removes the three extra operators:

| | Flexible batch | Batch fixed to 1 |
|---|---|---|
| Operators | 11 | 8 |
| File size | 21,736 bytes | 20,772 bytes |
| Outputs on 155 test inputs | — | identical predictions |

- **Laptop check, Keras vs TensorFlow Lite interpreter:** same prediction on 155/155 test inputs; largest probability difference 4.8e-7 (mean 1.6e-8); accuracy 91.6% for both (`results/fp32/conversion_check.json`).

## 3. Desktop vs on-device comparison (part C, needs the board — Moaz)
- **Saved for the comparison:**
  - Test inputs and labels: `results/fp32/test_inputs.npy`, `test_labels.npy`
  - Laptop outputs: `desktop_keras_outputs.npy`, `desktop_tflite_outputs.npy`
- **Still to fill in:** board outputs, prediction agreement, accuracy, and the explanation of any differences (summation order, fused multiply-add on the Cortex-M4F, the `exp()` implementation in softmax, input transfer precision).
