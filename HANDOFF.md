# Handoff to Moaz's Claude (Group 7, CSCE 4930 project, 2026-11-01)

Read first: `CSCE4930 - Fall 2026 - Project.pdf` (the brief), `AI_LOGBOOK.md` (every decision and why), `README.md`.

## Rules for the agent
- Group members decide; the agent proposes and waits for approval before acting.
- Record every question, suggestion and decision in `AI_LOGBOOK.md` under a new "Session — Moaz" heading, and add Moaz's work to the README "Contributions" table.
- No invented or synthetic data. Ziad rejected it (`AI_LOGBOOK.md` §1.13). Validation and test sets stay unchanged.
- Course FAQ: negative recordings are cut into 120 readings, skip 1, repeat, keep the short last piece; use only the brief's files (no lsj, negative_jiangyh, negative_ccw_*); augmentation is for training only; no external or newly collected recordings.
- This repository is only a transfer copy. Final work goes to the classroom repository under each member's own Git identity (commit history is graded per member).

## Done so far (Requirement 1 part A, plus the laptop half of part B)
- **Final FP32 model:** `results/fp32/fp32_model.keras`: 91.6% test (142/155), 81.5% validation. Known weaknesses (logbook §1.13): other seeds score 89.0–89.7%; validation "negative" is 17/33; a board lying still is classified as a gesture.
- **Converted model:** `results/fp32/model_fp32.tflite` (20,772 bytes, FP32, batch size fixed to 1). The same bytes as a C array are in `results/fp32/model_fp32_data.cc` (`g_model_fp32`, `g_model_fp32_len`, `alignas(16)`).
- **Laptop check:** Keras and the TensorFlow Lite interpreter agree on all 155 test predictions; largest probability difference 4.8e-7 (`results/fp32/conversion_check.json`).
- **Operator list:** `results/fp32/tflite_graph.md` lists 8 operators (5 types):
  - CONV_2D + RELU twice, MAX_POOL_2D twice
  - RESHAPE (Flatten)
  - FULLY_CONNECTED + RELU, FULLY_CONNECTED, SOFTMAX
  - The three Dropout layers are removed.
  - The pinned TensorFlow Lite Micro registers every operator and ignores operator versions.

- **Requirement 3, laptop part (done):** `results/fp32/model_cost.md`: parameters, storage and multiply-accumulates per operator (4,300 / 17,200 bytes / 62,016), and the `.tflite` size breakdown. Report-ready notes are in `report/`. Remaining for Moaz: the Flash share from the linker map, and cycles per operator.

## Model contract (the firmware must match it exactly)
- **Input:** float32 `[1, 128, 3, 1]`: 128 readings × (x, y, z) in **g**, i.e. raw milli-g **divided by 1000**.
- **Test windows:** the gesture sits at the end of the window. Earlier rows repeat the first reading (no noise). Recordings over 128 rows keep their first 128.
- **Output:** float32 `[1, 4]` softmax, in the order 0 wing, 1 ring, 2 slope, 3 negative.
- **Test data:**
  - `results/fp32/test_inputs.npy` (155 × 128 × 3 × 1, float32) and `test_labels.npy`
  - laptop outputs in `desktop_keras_outputs.npy` and `desktop_tflite_outputs.npy`
- **Tensor arena:** 61,440 bytes (60 KiB), fixed by the brief.

## Deviation to report, and a decision already made
- **TensorFlow version:** the brief pins TensorFlow / TensorFlow Lite Micro to commit 919f693 (TensorFlow 2.6.0). TensorFlow 2.6.0 cannot run on Ziad's Apple Silicon Mac (its build needs AVX instructions), so training and conversion used TensorFlow 2.21. The board side must still use the pinned TensorFlow Lite Micro.
  - If Moaz's computer is x86-64, the conversion can be redone in TensorFlow 2.6.0: rebuild the network with `training/model.py`, copy the weights layer by layer (conv1, conv2, dense1, dense2), check that the outputs match, then convert.
  - TensorFlow 2.6.0 cannot open Keras 3 `.keras` files, so export the weights to `.npz` from the 2.21 environment first.
- **Flatten (decided by Ziad, done):** with the batch size left flexible, Flatten became SHAPE, STRIDED_SLICE, PACK and RESHAPE. The model is now converted with a fixed batch of 1, so Flatten is a single RESHAPE; outputs are unchanged on all 155 test inputs. **Do not reconvert on your side.** The TensorFlow version question stays with Moaz.

## Moaz's tasks (the `fp32` tag and interview on 2 November cover Requirements 1–4)
1. **Set up the reference toolchain:**
   - GNU Arm Embedded 10-2020-q4-major, AmbiqSuite Rel2.2.0, FlatBuffers 1.12.0
   - TensorFlow at commit `919f693420e35d00c8d0a42100837ae3718f7927`
   - the linker script path from section 3 of the brief

   Flash a stock example (hello_world for the SparkFun Edge) first, to prove that the board, toolchain and flashing all work.
2. **Firmware:**
   - Load `g_model_fp32` with an op resolver that registers the 5 operator types above (CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX), and a 60 KiB arena.
   - Run each of the 155 test inputs.
   - Print the 4 probabilities over serial with 9 significant digits.
   - Send the inputs without rounding: either a const array in Flash (238,080 bytes) or raw float32 bytes over serial.
3. **Compare** the board outputs with `desktop_tflite_outputs.npy` and `desktop_keras_outputs.npy`: largest difference, prediction agreement and accuracy. Explain any differences (summation order, fused multiply-add on the Cortex-M4F, the `exp()` implementation, input transfer).
4. **Commit** the firmware, scripts and results, and add build, flash and reproduce steps to the README.
5. **Then Requirements 2–4:** memory map, multiply-accumulate counts and latency, roofline. These need the firmware build outputs (linker map) and the board's cycle counter.

## Running the laptop side (Python 3.12)
```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# data: extract http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz into data/
.venv/bin/python -m pytest                          # 27 tests
.venv/bin/python -m training.train_fp32             # retrains results/fp32 (seed 42; exact on the same TF and CPU type)
.venv/bin/python -m training.convert_fp32           # .tflite, C array, test arrays, laptop check
.venv/bin/python -m analysis.inspect_tflite_graph   # operator table; needs the pinned TensorFlow checkout in tensorflow/
.venv/bin/python -m analysis.plot_training results/fp32
```
