# 🧠 YOLO Modernization: From Darknet to PyTorch

## 🚀 Project Overview

This project modernizes the original YOLOv1 pipeline by migrating it from the legacy **Darknet (C)** implementation to a clean, efficient, and scalable **PyTorch-based** framework. The new codebase is designed for performance, parallelism, and research flexibility—enabling training across multiple GPUs and servers with ease.

---

## ✅ Key Achievements

### ⚙️ Parallelization & Scalability

- **Original Darknet** code (see `detection_layer.c`) relied heavily on **sequential loops** in C, which hindered GPU acceleration and parallel computation.
- This project rewrites critical sections (loss computation, prediction processing) using **vectorized PyTorch tensor operations**, achieving **full GPU parallelism**.
- Training and inference can now scale across **multiple GPUs and distributed nodes**.

### 🧮 Detection Layer & Loss Overhaul

- The YOLOv1 **loss function**, originally interwoven with post-processing logic in `detection_layer.c`, has been **decoupled and modularized**.
- Implemented entirely with tensor ops in `yolo_loss.py`, supporting:
  - Clean **object confidence**, **coordinate**, and **classification loss** terms.
  - Corrected inconsistencies from the original implementation (e.g., target assignment logic and confidence scaling).
- Enables **autograd support**, better performance, and improved readability.

### 🧰 Modern Deep Learning Stack

- **Project architecture** supports:
  - ✅ Easy configuration with `.toml` files (`yolo_config.toml`)
  - ✅ Logging & experiment tracking with **Weights & Biases (W&B)**
  - ✅ Checkpointing and resumable training
  - ✅ Scalable deployment across hardware backends
- Built for research and production alike.

### 🔍 Fidelity + Fixes

- While aligned with the **original YOLOv1 paper**, this implementation corrects **deviations** found in the Darknet code, including:
  - Misapplied confidence loss weights
  - Inefficient box assignment
- The result is a **more accurate, faithful, and optimized YOLOv1**.

---

## 📁 File Structure

```
├── main.py               # Training and evaluation entry point
├── voc_data.py           # Dataset and dataloader utilities (VOC-style)
├── yolo_model.py         # PyTorch YOLO model architecture
├── yolo_loss.py          # Vectorized YOLOv1 loss computation
├── yolo_utils.py         # IOU, NMS, and box transformation helpers
├── yolo_config.py        # Configuration schema
└── yolo_config.toml      # Editable experiment config file
```

---

## 🧪 Quick Start

```bash
# Run training using config
python main.py --config yolo_config.toml
```

Customize hyperparameters via `yolo_config.toml`.

---

## 📊 W&B Integration

Track experiments automatically with **Weights & Biases**:
- Loss curves
- Precision/mAP metrics
- Config snapshots

To enable:
1. Log in via `wandb login`
2. Training runs will be automatically tracked

---

## 🔄 Differences vs. Darknet

| Component             | Darknet (C)                     | This Project (PyTorch)              |
|----------------------|----------------------------------|-------------------------------------|
| Language             | C                                | Python                              |
| Loss Implementation  | For-loops, manual gradients      | Fully tensorized, autograd-enabled |
| Post-processing      | Sequential                       | Parallel, batched tensor ops        |
| Optimizer            | Custom Adam                      | PyTorch optimizers (AdamW, SGD)      |
| Config               | Hardcoded or `.cfg` style        | TOML-based, clean and extensible   |
| Logging              | Console print                    | W&B integrated                      |
| Scalability          | Single-threaded                  | Multi-GPU, distributed ready        |

---

## 📌 Notes

- The implementation remains close to **YOLOv1's original logic**, with select modernizations for:
  - Optimizers and schedulers
  - Logging and metrics
  - Data pipeline structure

- All changes aim to retain the **core functionality and spirit** of YOLOv1 while **bringing it up to modern ML standards**.

---

## 🙌 Acknowledgments

- [Joseph Redmon](https://github.com/pjreddie/darknet) for the original Darknet YOLO implementation
- The original [YOLOv1 paper](https://arxiv.org/abs/1506.02640)