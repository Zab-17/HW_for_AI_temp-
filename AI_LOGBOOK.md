# AI usage logbook — Group 7

**Course:** CSCE 4930 Hardware Architecture for AI and ML, Fall 2026
**Project:** Hardware-Aware ML Deployment on the SparkFun Edge
**Assistant used:** Claude (Anthropic), via Claude Code

Every question a group member puts to the assistant is recorded here in order, with a summary of the answer, the suggestions made, and the decision the group member took. Decisions are always made by the group member named in the entry; the assistant proposes and writes code only when asked.

---

## Session 1 — 2026-11-01 (Ziad)

### 1.1 What is the project about?
**Question:** Walk me through the problem, every requirement, every deliverable, every challenge, and what I am supposed to learn.
**Answer (summary):** A model trained on a laptop behaves differently on a microcontroller (about 1 MB Flash, 384 KB SRAM, Cortex-M4F). The project asks us to predict, from first principles, where every byte and cycle goes when the Magic Wand gesture network runs on the SparkFun Edge, then quantize it to 8-bit integers and apply two hardware-aware optimizations. Walked through Requirements 1–6, the deliverables (repository, README, PDF report, interviews), grading weights, milestones (FP32 part due 2 November with the `fp32` tag), and the main challenges (pinned 2020-era toolchain, moving test inputs to the board, accurate cycle counting, explaining every difference). Hand-calculated the parameter and multiply-accumulate counts: 4,300 parameters (17,200 bytes) and 62,016 multiply-accumulates; Dense 1 holds the most weights while Convolution 1 does the most work.

### 1.2 What does the model predict? Is it a visual model? Does it have labels and ground truth?
**Answer (summary):** It classifies a 128 × 3 window of accelerometer readings into wing, ring, slope or negative (no gesture). It is a motion model, not a visual one: it borrows 2-D convolution from image models because a gesture pattern can appear anywhere in the window. Labels come from the folder each recording sits in, one label per whole gesture; the folder name is the ground truth (the gesture the person was asked to perform).

### 1.3 Requirements 1, 2 and 3 explained simply; what is the tensor arena?
**Answer (summary):** Requirement 1 = train, deploy, list what the converter changed (Dropout removed, ReLU fused, Flatten → Reshape, Softmax separate), and compare laptop vs board on the same 155 test inputs. Requirement 2 = map Flash and SRAM; the arena is one fixed 60 KiB SRAM block from which TensorFlow Lite for Microcontrollers carves every intermediate result, reusing space once a result is dead (peak expected at Convolution 1: 13,824 bytes). Requirement 3 = parameter/multiply-accumulate counts, model file size vs weights, and measured cycles per layer.

### 1.4 What does one data point look like?
**Answer (summary):** Walked through the first ring by person `hyw` row by row (70 rows, 2.8 s). x, y, z are the pull along the board's three axes in thousandths of g; gravity makes a still board read about 1,000 in total, and how it is shared between axes shows tilt. One data point = one whole gesture (all its rows) plus one label, not one row. Rows arrive every 0.04 s; gestures average 3.1 s (wing 3.4 s, ring 3.2 s, slope 2.6 s); a gesture ends at the next `-,-,-` line.

### 1.5 How many examples per split? Who decided the split?
**Answer (summary):** 434 training / 92 validation / 155 test (681 total), fixed by the project brief and split by person so the score measures performance on new people. The assistant reproduced these counts exactly from the raw files. Noted the class imbalance: only 19 of 434 training examples are negatives (4.4%), against 36% of validation.

### 1.6 Padding: what is it, what are the options, can we cut duplicate rows?
**Answer (summary):** The network needs exactly 128 rows; 597 of 601 gestures are shorter, typically about 40% padding. Options compared: zeros (unrealistic: a still board never reads 0), repeating the edge reading, edge + small jitter, stretching to 128 rows; and gesture position (before / after / both). Only 4 gestures (all in training) exceed 128 rows. Cutting "duplicate rows" was checked on those 4: they contain no exactly repeated rows and no still rows at their edges; their near-still rows are pauses mid-gesture, and deleting them would squash time.

### 1.7 Research: which padding maximizes accuracy? (trusted sources only)
**Method:** Four research agents in two pairs with opposed lenses (benchmark studies vs accelerometer literature; original code vs a skeptic), primary sources only, with a source-quality log each.
**Findings (cross-matched):**
- Do not stretch gestures to 128 rows (4 of 4 agents): resampling scored 59% vs 63–68% for padding in the only head-to-head study (Iwana, ICASSP 2022, arXiv 2212.06536), and the board cannot stretch a live stream.
- The original TensorFlow Magic Wand scripts pad with the edge reading plus ±10 uniform noise, two copies (before and after), keep the first 128 rows of long gestures, and do not scale inputs (verified in `train/data_load.py` at the pinned commit).
- Training with the gesture at several positions matches the board's sliding window.
- Jitter does not raise accuracy by itself (Um et al., ICMI 2017, arXiv 1706.00527: 77.52% vs 77.54%).
- Zero vs edge padding: the only measured comparison favoured zero padding on min-max-normalized data; on raw thousandths-of-g zero means free fall. Ruled edge padding, medium confidence.
- Sources rejected as non-primary: Medium, Towards Data Science, GeeksforGeeks, Analytics Vidhya, Kaggle notebooks, Hackaday/Elektor marketing.

**Decision (Ziad):** training = edge reading + ±10 jitter, two copies (before and after); validation and test = one deterministic copy padded before (no jitter), keeping 92 / 155 unchanged; long gestures keep their first 128 rows.

### 1.8 Course FAQ check
**Question:** Does the FAQ contradict our decisions?
**Answer (summary):** No. The FAQ's negative-chunking rule, file lists and "validation and test unchanged" rule match our pipeline exactly. The FAQ also allows synthetic and augmented training data, including negatives — a possible fix for the 19-negative imbalance.

### 1.9 Training settings
**Answer (summary):** Explained epochs, learning rate, batch size, early stopping, which checkpoint to keep (by validation, never test), class weights and random seed using an exam-preparation analogy.
**Decision (Ziad):** Adam, learning rate 0.001, batch size 32, up to 100 epochs with early stopping on validation accuracy (patience 15, best weights restored), balanced class weights, seed 42, no input scaling. Also asked for the gradient of every layer to be tracked and graphed after the first run.

### 1.10 First FP32 training run — failed to learn
**Code written by the assistant on request:** `training/dataset.py`, `training/model.py`, `training/gradient_tracking.py`, `training/train_fp32.py`, `analysis/plot_training.py`, `tests/` (13 tests: split counts 434/92/155, 4,300 parameters, padding rules), `pytest.ini`.
**Result:** test accuracy 49.7%, validation 54.3% (best epoch 2 of 17). The gradient record showed 27 of 476 batches where both convolutions and Dense 1 received exactly zero gradient, and spikes up to about 1,860 — dead ReLU neurons caused by raw inputs of up to ±2,000.
**Diagnosis (assistant, in a scratch copy):** Keras's built-in training step reproduced the same 49.7%, so the gradient-tracking code is not at fault (it only failed to display the training loss). Dividing inputs by 1,000 raised accuracy to 75–85%, still below the 90% target, with signs of memorizing the training people.
**Status:** fixes proposed to Ziad (see 1.11).

### 1.11 Finish the training, fixing issues along the way
**Instruction (Ziad):** finish the whole training, fix the issues along the way, and report only at the end.
**What the assistant did, run by run** (every change was chosen from validation results and a still-board check; test accuracy is listed for the record):

| Run | Change | Validation | Test | Finding |
|---|---|---|---|---|
| 1 | settings of 1.9, raw inputs | 54.3% | 49.7% | dead ReLU neurons from ±2,000 inputs |
| 2 | inputs divided by 1,000 (units of g); training-loss logging fixed | 79.3% | 75.5% | gestures fine, no-gesture class 16/33 on validation |
| 3 | + original augmentation (7 speed, 7 size, 5 offset variants per recording) + overlapping 120-row windows every 10 rows of negative_3 | 81.5% | 91.6% | fragile: seeds 43/44 gave 89.0% / 89.7% test; a still board was called a gesture in 4 of 5 orientations |
| 4 | + 1,000 synthetic no-gesture windows (500 still boards, 500 slow tilts, ±20 milli-g tremor, as in the original scripts) | **94.6%** | **93.5%** | no-gesture 29/33 on validation; a still board is "no gesture" in 48 of 50 random orientations |

**Final model at that point:** run 4 with seed 42 (the seed fixed before the first run), saved as `results/fp32/fp32_model.keras`. **Reversed by Ziad in 1.13: the invented examples were removed and run 3 became the final model.** Re-running the default command reproduced it exactly (identical per-batch gradient record). 27,800 training windows after augmentation and padding; validation and test unchanged at 92 / 155.
**Robustness check (same settings, other seeds, not used for selection):** test 89.0% (seed 43), 91.0% (seed 44), 93.5% (seed 42); validation 94.6–95.7% for all three. Seed 43 stopped at epoch 9 because validation accuracy on 92 examples moves in coarse 1.1-point steps and plateaued early.
**Honesty note:** test accuracy was printed during the diagnosis of runs 1–3, so the test set is not perfectly untouched; no setting was chosen by its test score.
**Code added:** `training/augmentation.py` (+ tests), input scaling `to_g` in `training/dataset.py`, loss-logging fix in `training/gradient_tracking.py` (+ tests), run names and augmentation settings in `training/train_fp32.py`, per-epoch gradient bands in `analysis/plot_training.py`. 23 tests pass.
**Firmware consequence:** the board must divide every accelerometer reading by 1,000 before inference.

### 1.12 Questions about the runs
**Questions (Ziad):** walk me through the runs very concisely (what changed, why, how it affected the result, why, and how many epochs); what does "flat" mean; explain runs 3 and 4 in simpler terms; why label a flat window "no gesture"; did the original data contain a still board labelled "no gesture"?
**Answers (summary):**
- Epochs per run (each run stops 15 epochs after its best validation epoch): run 1 = 17 (best 2), run 2 = 23 (best 8), run 3 = 35 (best 20), run 4 = 57 (best 42); 132 in total, plus the robustness reruns.
- "Flat" means consecutive rows with the same numbers. The padding is flat (the last reading repeated), so about 40% of every gesture window is flat, while the real no-gesture windows are hand movement almost all the way through.
- Run 3's model therefore learned the cue "long flat stretch means gesture": shown a board lying completely still (128 identical gravity-only rows) in 5 positions, it answered slope, ring, ring, no gesture, slope.
- Run 4 added computer-generated still and slowly-turning boards labelled "no gesture" so that flat stretches appear in both classes.
- Measured on request: the course data contains **no** still-board no-gesture example. The longest still stretch in any negative recording is 11 rows (0.4 s), and none of 886 overlapping 120-row windows (one starting every 10 rows) across negative_1–5 is calm. The original TensorFlow scripts generated still-board negatives by code; the course's 434 / 92 / 155 counts exclude them.

### 1.13 Objection and decision: remove the invented examples
**Ziad's objection:** the assistant created new data points and assigned them an existing label ("no gesture"), which Ziad had not approved.
**Assistant's response:** confirmed that run 4 invented 1,000 training examples that come from no real recording and labelled them "no gesture"; acknowledged that this was a significant change to the training data made without asking first, and that it should have been raised with Ziad before running it. Noted that the course FAQ allows synthetic training data, and that the label rests on the assistant's assumption that a still board draws no gesture, which the validation and test sets cannot confirm because they contain no still boards. Offered three options: remove them and return to run 3; keep run 4 and disclose it; or try a real-data-only alternative (shorter random-length negative pieces padded like gestures).
**Decision (Ziad):** remove the invented examples and go back to run 3.
**Done:**
- Deleted the generating code (`synthetic_still_board`, `synthetic_slow_tilt` and their helpers in `training/augmentation.py`, `synthetic_negatives` and its setting in `training/train_fp32.py`) and their two tests; 21 tests pass.
- Retrained with the default command: the result is identical to run 3 (same per-batch gradient record). **Final model: `results/fp32/fp32_model.keras` — 91.6% test (142 / 155), 81.5% validation, 99.9% training; 35 epochs, best at epoch 20; 25,800 training windows, all derived from real recordings.**
- Known limitations of the final model, accepted with the decision: other seeds score 89.0% and 89.7% on test; validation no-gesture is 17 / 33; a still board is classified as a gesture in 4 of 5 orientations.
- Run 4's result folders (`results/fp32_run4_*`) were deleted at Ziad's request; the run is documented only in this logbook, and the final pipeline contains no invented data.

### 1.14 Following the project description: reference TensorFlow version
**Instruction (Ziad):** follow the project description (`CSCE4930 - Fall 2026 - Project.pdf`).
**Finding (assistant):** section 3 of the brief pins the reference environment to TensorFlow / TensorFlow Lite Micro commit 919f693 (TensorFlow 2.6.0); training had used TensorFlow 2.21. Everything else in Requirement 1 part A matched the brief. Remaining parts of Requirement 1: deploy on the SparkFun Edge, characterize the deployed graph, compare desktop and on-device outputs.
**Options offered:** (1) keep the trained weights and convert with TensorFlow 2.6.0; (2) port training to 2.6.0 and retrain; (3) convert with 2.21 and only check compatibility.
**Decision (Ziad):** option 1.
**Result:** TensorFlow 2.6.0 cannot run on Ziad's MacBook (Apple M1 Pro). Only Intel builds exist for macOS; under Rosetta it stops immediately with "The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine." A different machine is needed for option 1.

### 1.15 Hand-off to Moaz, who has the board
**Instruction (Ziad):** push the work to `https://github.com/Zab-17/HW_for_AI_temp-.git` so Moaz can pull it and work with the board.
**Done:** one commit containing all 54 project files (code, tests, results, README, this logbook, the brief), authored and committed as Zab-17 with Claude as co-author, pushed to `main`. The two earlier repository-skeleton commits by Moaz were not pushed, because Ziad's rule allows only Zab-17 and Claude as contributors on his repositories; Moaz's skeleton work is credited in the README contributions table. The classroom repository was not touched.

### 1.16 Option (b): convert without the board, write a handoff, push
**Instruction (Ziad):** write a compact handoff file for Moaz's Claude with what was done and what he should do; do option (b) since it needs no board; then push.
**Done:**
- `training/convert_fp32.py`: converts `results/fp32/fp32_model.keras` with TensorFlow 2.21 (no quantization) to `results/fp32/model_fp32.tflite` (21,736 bytes) and a C array; saves the 155 test inputs, labels and laptop outputs. Keras and the TensorFlow Lite interpreter agree on all 155 predictions (largest probability difference 4.8e-7; both 91.6%).
- `analysis/inspect_tflite_graph.py` → `results/fp32/tflite_graph.md`: 11 operators. ReLU is fused into both CONV_2D and the first FULLY_CONNECTED; the three Dropout layers are removed; Softmax is a separate operator; Flatten became SHAPE → STRIDED_SLICE → PACK → RESHAPE because the batch size was left flexible. Every operator is registered in the pinned TensorFlow Lite Micro `all_ops_resolver.cc`, and the pinned resolver ignores operator versions.
- Open decision for the group: convert with a fixed batch of 1 to reduce Flatten to a single RESHAPE.
- `HANDOFF.md`: state, model contract, deviation, and Moaz's tasks. 27 tests pass.

### 1.17 Moaz's review; TensorFlow version and Flatten handed to Moaz
**Context:** Moaz's assistant merged the temp repository into the group repository and listed open items: the TensorFlow version (the brief pins 2.6.0; 2.21 was used) and the Flatten conversion (4 operators instead of 1). Moaz chose to keep TensorFlow 2.21 and report the deviation, and to convert with a fixed batch of 1.
**Questions (Ziad):** what is the Flatten question and which requirement is it; what is the problem; why use 2.21.
**Answers (summary):** Flatten turns the 14 × 16 block of numbers into one line of 224 for the dense layer. The flexible batch size made the converter add SHAPE, STRIDED_SLICE and PACK to compute the line length at run time. This is part of Requirement 1 ("operators that are fused, removed, or represented differently"); the brief asks for it to be explained, not fixed. Tested in scratch: rebuilding the network with `keras.Input(batch_shape=(1, 128, 3, 1))`, copying the trained weights and converting gives 8 operators and 20,624 bytes, with outputs identical on all 155 test inputs. A first attempt through `tf.function(...).get_concrete_function(...)` stored the weights as variables (19 extra operators, wrong outputs) and must not be used. TensorFlow 2.21 was used because 2.6.0 cannot run on Apple Silicon (AVX); the assistant had installed 2.21 without first checking section 3 of the brief.
**Decision (Ziad):** Moaz sorts out both the TensorFlow version and the Flatten conversion. The assistant does not change the model.

### 1.18 Flatten fixed here after all
**Decision (Ziad):** "we do the Flatten question" (reverses the Flatten part of 1.17; the TensorFlow version stays with Moaz).
**Questions (Ziad):** what is the Flatten question; what was done and why, simply.
**Answer (summary):** the board runs a list of instructions. Flatten only rearranges 224 numbers from a 14 × 16 grid into one line, which needs one RESHAPE. Because the model was saved to accept any number of recordings at once, the converter added SHAPE, STRIDED_SLICE and PACK to work out "how many recordings?" on every run. The board always receives exactly one, so those three always compute 1.
**Done:** `training/convert_fp32.py` now rebuilds the network with a fixed batch of 1 (`with_fixed_batch`, same layers and weights, with a test) before converting. Result: 8 operators instead of 11, `model_fp32.tflite` 20,772 bytes instead of 21,736, Keras and TensorFlow Lite agree on all 155 test predictions (largest difference 4.8e-7, 91.6% both). `results/fp32/tflite_graph.md`, the C array and the laptop outputs were regenerated; `HANDOFF.md` updated. 28 tests pass.

### 1.19 Records for the report; Requirement 3 laptop part
**Instruction (Ziad):** record everything for the report, and do the next task that does not need the board.
**Done:**
- `report/notes_requirement1.md` and `report/notes_requirement3.md`: report-ready facts and tables.
- `analysis/tflite_model.py`: a small read-only view of a `.tflite` file, shared by the analysis scripts. `analysis/inspect_tflite_graph.py` now uses it, with unchanged output.
- `analysis/model_cost.py` → `results/fp32/model_cost.md` / `.json`: parameters, storage and multiply-accumulates per operator. Totals: 4,300 parameters, 17,200 bytes, 62,016 multiply-accumulates, matching the hand calculation, with tests.
  - Largest storage is dense1 (14,400 bytes); largest work is conv1 (36,864), explained by weight reuse (384 uses per conv1 weight, 1 per dense weight).
  - The 20,772-byte file breaks down as 17,200 bytes of parameters plus 3,572 bytes of graph description (names, structure, metadata, padding).
- 30 tests pass. The parts of Requirement 3 that need the firmware build or the board (Flash share, cycles) remain with Moaz.

### 1.20 Planning questions (recorded on request)
**Questions (Ziad) and answers (summary):**
- *Did you convert to TensorFlow Lite?* Yes: `results/fp32/model_fp32.tflite`, plus a C array for the firmware. It was converted with TensorFlow 2.21 (deviation from the brief's 2.6.0, see §1.14), and its predictions match Keras on all 155 test inputs.
- *What is the next step?* Running the model on the board, which is Moaz's part. The `fp32` tag and interview on 2 November cover Requirements 1–4, so the board-free parts should be done in parallel.
- *What did you tell Moaz to do?* The five tasks in `HANDOFF.md`: reference toolchain and a stock example; firmware running the 155 test inputs; board-vs-laptop comparison; commit with README steps; then the measurements for Requirements 2–4. Also the rules: no invented data, log AI use, commit under his own name.
- *What can I do in parallel?* Board-free tasks, in suggested order:
  1. the Flatten decision (done, §1.18);
  2. a comparison script for the board's outputs;
  3. Requirement 3 counts (done, §1.19);
  4. Requirement 2 theoretical memory per operator;
  5. Requirement 4 theory (compute-bound or memory-bound per operator).

  Also possible on the laptop, untested: the pinned library's arena plan, and a firmware build for the memory map.

### 1.21 Second handoff and push
**Instruction (Ziad):** write a new handoff for Moaz and push.
**Done:** `HANDOFF_2.md`, covering:
- the changes since `HANDOFF.md` (Flatten fixed, do not reconvert; TensorFlow version left to Moaz; Requirement 3 laptop part; report notes);
- how to merge with Moaz's uncommitted edits (expected conflicts in `AI_LOGBOOK.md` and `README.md`, keep both sides);
- Moaz's tasks, including a proposed `RESULT,<index>,<four probabilities>` line format for the board output, and the warning that `test_inputs.npy` is already in g;
- what Ziad does in parallel.

The work is pushed to the temp repository as a commit authored by Zab-17.
