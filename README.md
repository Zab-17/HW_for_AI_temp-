# CSCE 4930 – Hardware-Aware ML Deployment on the SparkFun Edge

Group 7 – Fall 2026

Gesture recognition (Magic Wand accelerometer dataset) on the SparkFun Edge
(Ambiq Apollo3 Blue, Arm Cortex-M4F) using TensorFlow Lite Micro.

## Repository layout

| Path | Contents |
|------|----------|
| `data/` | Magic Wand dataset (not committed; see `data/README.md`) |
| `training/` | Dataset preparation, FP32 / QAT / PTQ training and TFLite conversion |
| `firmware/` | TFLM application for the SparkFun Edge |
| `analysis/` | Measurement and analysis scripts (memory, MACs, latency, roofline) |
| `results/` | Raw measurements and generated tables/figures |
| `report/` | Project report (PDF) |

*(Files will be listed here with a brief description as they are added.)*

## Reference environment

- TensorFlow / TFLM commit `919f693420e35d00c8d0a42100837ae3718f7927`
- GNU Arm Embedded Toolchain 10-2020-q4-major
- AmbiqSuite Rel2.2.0
- FlatBuffers 1.12.0

## Build, run, reproduce

TODO – to be filled in as each milestone is completed.

## Milestones

- [ ] FP32 deployment and analysis (Requirements 1-4) – tag `fp32`, due Nov 2
- [ ] INT8 deployment and comparison (Requirement 5)
- [ ] Hardware-aware optimizations (Requirement 6)
- [ ] Final submission – tag `final`, due Dec 6

## Team

- Moaz Allam
- Zeyad Maher
- Wahiba

## Contributions

Who did what, updated as work lands. Questions put to the AI assistant and its
suggestions are logged in [`AI_LOGBOOK.md`](AI_LOGBOOK.md).

| Date | Member | Work |
|------|--------|------|
| 2026-11-01 | Moaz | Created the repository: README skeleton, `.gitignore`, data notes, team list |
| 2026-11-01 | Ziad | Analysed the dataset (file format, labels, gesture lengths, split counts) and checked it against the brief and FAQ |
| 2026-11-01 | Ziad | Chose the padding and training settings after an AI-assisted literature and code review (see `AI_LOGBOOK.md` §1.7–1.9) |
| 2026-11-01 | Ziad | Data pipeline, baseline model, gradient tracking, FP32 training and plotting scripts, tests (`training/`, `analysis/`, `tests/`; written with AI assistance) |
| 2026-11-01 | Ziad | First FP32 training run (`results/fp32_run1_raw_inputs/`): 49.7% test accuracy; diagnosed as raw input scale causing dead ReLU neurons |
| 2026-11-01 | Ziad | Fixed input scaling and added training augmentation from real recordings only; final FP32 model `results/fp32/`: 91.6% test / 81.5% validation (AI-assisted, see `AI_LOGBOOK.md` §1.11) |
| 2026-11-01 | Ziad | Reviewed an AI-proposed run with 1,000 computer-generated no-gesture examples and rejected it; invented data removed from the pipeline (`AI_LOGBOOK.md` §1.13) |
