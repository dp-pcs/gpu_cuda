# GPU/CUDA Hands‑On Tutorial Project

This repo teaches you—step by step—how to *see and measure* the difference between **CPU**, **Apple MPS (Metal)**, and **NVIDIA CUDA** for deep learning.

---

## 🚀 Quick Setup (Choose Your Platform)

Pick the setup guide for your environment:

### 📱 [Google Colab Setup](SETUP_GOOGLE_COLAB.md)
**Best for:** Free GPU access, no local setup required
- ✅ Free NVIDIA GPU (T4, V100, or A100)
- ✅ Pre-installed CUDA drivers
- ✅ No installation needed

### ☁️ [AWS / Paperspace Setup](SETUP_AWS_PAPERSPACE.md)
**Best for:** Dedicated GPU instances, production workflows
- ✅ Scalable GPU instances (g4dn, g5, p3, A100)
- ✅ Docker support with NVIDIA Container Toolkit
- ✅ Full CUDA and TensorRT support

### 🍎 [Mac Local Setup (Apple Silicon)](SETUP_MAC_LOCAL.md)
**Best for:** Learning on M1/M2/M3/M4 Macs
- ✅ Apple MPS (Metal Performance Shaders) GPU acceleration
- ✅ Local development environment
- ✅ 2-6x speedup over CPU

> **Note:** CUDA requires NVIDIA GPUs. For Mac users, we use MPS (Apple's GPU framework) instead.

---

## What you’ll do
- **Train a tiny CNN** on MNIST and time CPU vs GPU/MPS runs.
- **Profile** a short run with PyTorch Profiler and `nvidia-smi`.
- **Run a tiny CUDA kernel** (vector add) and see blocks/threads in action.
- **(CUDA only)** Export to ONNX and run a **TensorRT** inference benchmark.

Results are saved to `results/metrics.csv` and a plot PNG.

---

## Commands (Makefile)
```bash
make verify
make train-cpu
make train-cuda
make train-mps
make export-onnx
make trt-benchmark      # CUDA + TensorRT only
```

---

## Troubleshooting

For platform-specific issues, see the detailed troubleshooting sections in:
- [Google Colab Troubleshooting](SETUP_GOOGLE_COLAB.md#troubleshooting)
- [AWS/Paperspace Troubleshooting](SETUP_AWS_PAPERSPACE.md#troubleshooting)
- [Mac Local Troubleshooting](SETUP_MAC_LOCAL.md#troubleshooting)

**Common issues:**
- **Out of memory (OOM)**: Lower `--batch-size` (e.g., 128 → 64 → 32 → 16)
- **GPU not detected**: See platform-specific setup guides above
- **Import errors**: Make sure virtual environment is activated

---

## Learn the mental model (TL;DR)
- **CUDA** lets you run many lightweight **threads** organized into **blocks** (which form a **grid**) on the GPU.
- Warps (groups of 32 threads) execute together; avoid divergent branching.
- **Shared vs global memory**: prefer shared for data reused by a block.
- DL frameworks (PyTorch/TensorFlow) use CUDA/cuDNN to accelerate ops.
