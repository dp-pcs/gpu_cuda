# GPU/CUDA Hands‑On Tutorial Project

This repo teaches you—step by step—how to *see and measure* the difference between **CPU**, **Apple MPS (Metal)**, and **NVIDIA CUDA** for deep learning.

> If you're on Apple Silicon: use the **MPS** path. CUDA requires an NVIDIA GPU (Colab/AWS/Paperspace/PC).

---

## Quickstart (choose one)

### 1) Google Colab (CUDA)
1. Runtime → Change runtime type → **GPU**.
2. Install CUDA wheels for PyTorch:
   ```python
   %pip install --upgrade pip
   %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   %pip install matplotlib onnx onnxruntime
   ```
3. Run: `!python src/train.py --device cuda --epochs 1 --batch-size 128`

### 2) AWS g5/Paperspace (CUDA)
```bash
# NVIDIA container toolkit required; then build & run
docker build -f env/Dockerfile.cuda -t cuda-tutorial .
docker run --gpus all -it -p 8888:8888 -v $PWD:/workspace cuda-tutorial bash

# inside container
python src/train.py --device cuda --epochs 1 --batch-size 128
```

### 3) Mac (Apple Silicon, MPS)
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install matplotlib onnx onnxruntime
python src/train.py --device mps --epochs 1 --batch-size 128
```

---

## Verify your accelerator
```bash
python -c "import torch; print('cuda?', torch.cuda.is_available()); print('mps?', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()); print('device count', torch.cuda.device_count())"
```

CUDA: also run `nvidia-smi` and watch utilization during training.

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
- **CUDA mismatch**: ensure driver/CUDA version matches your PyTorch wheels.
- **OOM (out of memory)**: lower `--batch-size` (e.g., 64 → 32 → 16).
- **MPS not available**: update macOS + PyTorch; fall back to CPU.
- **TensorRT missing**: skip TRT steps or use the Dockerfile.cuda image.

---

## Learn the mental model (TL;DR)
- **CUDA** lets you run many lightweight **threads** organized into **blocks** (which form a **grid**) on the GPU.
- Warps (groups of 32 threads) execute together; avoid divergent branching.
- **Shared vs global memory**: prefer shared for data reused by a block.
- DL frameworks (PyTorch/TensorFlow) use CUDA/cuDNN to accelerate ops.
