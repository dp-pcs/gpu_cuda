# Google Colab Setup Guide

**GPU/CUDA Hands-On Tutorial for Google Colab**

This guide walks you through running GPU-accelerated deep learning experiments on Google Colab with NVIDIA CUDA.

---

## Prerequisites

- Google account
- Access to Google Colab (free tier includes GPU access)

---

## Setup Instructions

### 1. Upload this repository to Google Drive or clone it in Colab

```python
# Option 1: Clone from GitHub
!git clone https://github.com/dp-pcs/gpu_cuda.git
%cd gpu_cuda

# Option 2: Upload files manually to your Google Drive and navigate to the folder
```

### 2. Change Runtime to GPU

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** as Hardware accelerator
3. Click **Save**

### 3. Install Dependencies

Run this in a Colab cell:

```python
%pip install --upgrade pip
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
%pip install matplotlib onnx onnxruntime
```

### 4. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
```

You can also check GPU utilization:
```python
!nvidia-smi
```

---

## Running Experiments

### Train CNN (CPU vs CUDA comparison)

```python
# CPU baseline
!python src/train.py --device cpu --epochs 1 --batch-size 128

# GPU accelerated
!python src/train.py --device cuda --epochs 1 --batch-size 128
```

### Profile GPU Usage

```python
!python src/profiler.py --device cuda
```

### Watch GPU utilization in real-time

In one cell, run:
```python
!watch -n 1 nvidia-smi
```

In another cell, run your training:
```python
!python src/train.py --device cuda --epochs 2 --batch-size 256
```

### Export to ONNX and Benchmark

```python
# Export trained model to ONNX format
!python src/export_onnx.py

# Run TensorRT inference benchmark (if available)
!python src/infer_trt.py
```

---

## What You'll Learn

- **Train a tiny CNN** on MNIST and compare CPU vs CUDA performance
- **Profile** GPU utilization with PyTorch Profiler and `nvidia-smi`
- **Export models** to ONNX format for deployment
- **Benchmark inference** with TensorRT (if available)

Results are saved to `results/metrics.csv` and plot images.

---

## Available Make Commands

If you prefer using the Makefile:

```python
!make verify          # Verify CUDA setup
!make train-cpu       # Train on CPU
!make train-cuda      # Train on CUDA
!make export-onnx     # Export to ONNX
!make trt-benchmark   # TensorRT benchmark
```

---

## Troubleshooting

### CUDA not available
- Verify you selected **GPU** runtime (Runtime → Change runtime type)
- Restart runtime if needed
- Check quota limits (Colab free tier has usage limits)

### Out of Memory (OOM)
- Lower `--batch-size`: try 64, 32, or 16
- Restart runtime to clear GPU memory
- Use `torch.cuda.empty_cache()` between experiments

### CUDA version mismatch
- Use the PyTorch wheels with the correct CUDA version (cu121 for CUDA 12.1)
- Check your runtime's CUDA version: `!nvcc --version`

### TensorRT not available
- TensorRT may not be pre-installed in Colab
- You can skip TensorRT benchmarks or install it separately
- Focus on PyTorch CUDA acceleration instead

---

## Understanding CUDA (Mental Model)

- **CUDA** lets you run many lightweight **threads** organized into **blocks** (which form a **grid**) on the GPU
- **Warps** (groups of 32 threads) execute together; avoid divergent branching for best performance
- **Memory hierarchy**: prefer shared memory for data reused by a block; global memory is slower
- Deep learning frameworks (PyTorch/TensorFlow) use CUDA/cuDNN to automatically accelerate operations

---

## Next Steps

1. Run the notebooks in the `notebooks/` directory
2. Experiment with different batch sizes and architectures
3. Compare training times between CPU and GPU
4. Try the custom CUDA kernel example (vector addition)

**Happy GPU computing! 🚀**

