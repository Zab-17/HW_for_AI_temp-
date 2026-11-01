# Handoff 2 to Moaz's Claude (2026-11-01, late evening)

Read `HANDOFF.md` first; this file lists only what changed since then and what to do now.

## What changed
1. **Flatten is fixed here; do not reconvert on your side.** Ziad decided to do it (`AI_LOGBOOK.md` §1.18).
   - The model is now converted with a fixed batch of 1 (`with_fixed_batch` in `training/convert_fp32.py`).
   - The board runs **8 operators of 5 types**: CONV_2D, MAX_POOL_2D, CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX.
   - `model_fp32.tflite` is 20,772 bytes. Its outputs are identical on all 155 test inputs.
   - Regenerated: `model_fp32.tflite`, `model_fp32_data.cc`, `desktop_*.npy`, `conversion_check.json`, `tflite_graph.md`.
2. **TensorFlow version:** left to Moaz (Moaz chose: keep 2.21 and report the deviation). If the conversion is ever redone in 2.6.0, keep the fixed batch:
   - Build with `keras.Input(batch_shape=(1, 128, 3, 1))` and copy the weights.
   - Do **not** convert through `tf.function(...).get_concrete_function(...)`. That route stored the weights as variables, added 19 operators and gave wrong outputs.
3. **Requirement 3, laptop part, done:** `results/fp32/model_cost.md`:
   - 4,300 parameters, 17,200 bytes, 62,016 multiply-accumulates.
   - Per-operator table, and the `.tflite` size breakdown.
4. **Report notes:** `report/notes_requirement1.md` and `report/notes_requirement3.md`.
5. **Code:** `analysis/tflite_model.py` (shared `.tflite` reader) and `analysis/model_cost.py`; 30 tests pass.
6. **Name:** the README team list now says "Ziad Maher" (your edit). Ziad has not confirmed the spelling yet.

## Merging this into the group repository
1. Commit your local edits first: the "Session — Moaz" logbook entries, your README row, the name fix.
2. Then pull: `git pull https://github.com/Zab-17/HW_for_AI_temp-.git main`.
3. Expect conflicts in `AI_LOGBOOK.md` and `README.md`, because both sides appended to the end of the same sections. **Keep both sides**: Ziad's §1.14–1.20 and your M entries; every Contributions row.
4. Keep the Zab-17 commits as they are, since commit history is graded per member.

## Moaz's tasks now (same order as before)
1. **Toolchain and a stock example on the board.** This is the blocker. The board was not detected: the SparkFun Edge needs a USB-serial adapter (CH340 or FTDI) and its driver, after which a new COM port appears.
2. **Firmware:**
   - Load `g_model_fp32` and register the 5 operator types; use a 61,440-byte arena.
   - Run the 155 inputs from `results/fp32/test_inputs.npy`. These are **already in g**, so do not divide them again. Only live accelerometer readings in milli-g need the division by 1000.
   - Print one line per input in this proposed format, so Ziad's comparison script can read it:
     `RESULT,<index>,<p_wing>,<p_ring>,<p_slope>,<p_negative>` with 9 significant digits.
     If you prefer another format, write it here so both sides match.
3. **Compare** with `desktop_tflite_outputs.npy` and `desktop_keras_outputs.npy`: largest difference, prediction agreement, accuracy, and explanations.
4. **Keep the build outputs Requirements 2–3 need:**
   - the linker `.map` file,
   - `interpreter.arena_used_bytes()`,
   - cycles per operator (Cortex-M4 DWT cycle counter or the TensorFlow Lite Micro profiler), with the clock settings written down.
5. **Then the Requirement 2–4 measurements.**

## Ziad works in parallel (no board)
- Requirement 2 theory: live activation memory per operator, and the peak.
- Requirement 4 theory: compute-bound or memory-bound per operator.
- The comparison script that reads the `RESULT` lines.
