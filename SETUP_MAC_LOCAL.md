# Mac Local Setup Guide (Apple Silicon)

**GPU/CUDA Hands-On Tutorial for Apple Silicon Macs**

This guide covers setup for Apple Silicon Macs (M1, M2, M3, M4) using **Apple MPS (Metal Performance Shaders)** for GPU acceleration.

> **Note:** NVIDIA CUDA is not available on Mac. This guide uses **MPS** (Metal Performance Shaders), Apple's GPU acceleration framework for PyTorch.

---

## Prerequisites

- Mac with Apple Silicon (M1, M2, M3, or M4 chip)
- macOS 12.3 or later (for MPS support)
- Python 3.9 or later
- Xcode Command Line Tools

### Install Xcode Command Line Tools (if needed)

```bash
xcode-select --install
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:dp-pcs/gpu_cuda.git
cd GPU_CUDA
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Troubleshooting:** If `python3 -m venv` fails, try:
```bash
# Use system Python
/usr/bin/python3 -m venv .venv

# Or install virtualenv
pip3 install --user virtualenv
python3 -m virtualenv .venv
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install matplotlib onnx onnxruntime
```

### 4. Verify MPS Access

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

You should see:
```
MPS available: True
MPS built: True
```

---

## Running Experiments

### Train CNN (CPU vs MPS comparison)

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# CPU baseline
python src/train.py --device cpu --epochs 1 --batch-size 128

# MPS accelerated (Apple GPU)
python src/train.py --device mps --epochs 1 --batch-size 128
```

### Profile MPS Usage

```bash
python src/profiler.py --device mps
```

### Export Model to ONNX

```bash
python src/export_onnx.py
```

### Run Jupyter Notebooks

```bash
pip install jupyter
jupyter notebook
```

Then navigate to the `notebooks/` directory and open:
- `01_verify_gpu.ipynb` - Verify MPS setup
- `02_train_cnn_cpu_vs_gpu.ipynb` - Compare CPU vs MPS performance
- `03_basic_cuda_kernel.ipynb` - Learn CUDA concepts (theory)
- `04_tensorrt_inference.ipynb` - ONNX export (TensorRT is CUDA-only)

---

## Using Makefile Commands

```bash
make verify       # Verify MPS setup
make train-cpu    # Train on CPU
make train-mps    # Train on MPS (Apple GPU)
make export-onnx  # Export to ONNX
```

**Note:** CUDA-specific commands (`make train-cuda`, `make trt-benchmark`) won't work on Mac.

---

## Troubleshooting

### MPS not available

**Check macOS version:**
```bash
sw_vers
```
MPS requires macOS 12.3 or later.

**Update PyTorch:**
```bash
pip install --upgrade torch torchvision torchaudio
```

**Fall back to CPU:**
If MPS is unavailable, use `--device cpu` instead.

### Virtual Environment Creation Failed

If you encountered the error: `source: no such file or directory: .venv/bin/activate`

This happened because the `venv` creation silently failed. Try:

```bash
# Option 1: Use system Python
/usr/bin/python3 -m venv .venv
source .venv/bin/activate

# Option 2: Use virtualenv instead
pip3 install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
```

### Import Error: "module 'profile' has no attribute 'run'"

This was caused by a naming conflict with `src/profile.py`. **This has been fixed** - the file is now named `src/profiler.py`.

If you still see this error:
- Delete any `profile.py` or `profile.pyc` files in the `src/` directory
- Make sure you're using `python src/profiler.py` (not `profile.py`)

### Out of Memory (OOM)

Apple Silicon has unified memory shared between CPU and GPU. If you run out of memory:

- Lower `--batch-size`: try 64, 32, or 16
- Close other applications to free up memory
- Use Activity Monitor to check memory pressure

### Slow Performance on MPS

- Make sure you're not running in Low Power Mode
- Close other memory-intensive applications
- Some operations may fall back to CPU (check profiler output)
- Larger batch sizes often perform better on MPS

---

## Performance Expectations

### M1/M2/M3 Performance

Typical speedup over CPU for deep learning:
- **M1**: 2-4x faster than CPU
- **M2**: 3-5x faster than CPU  
- **M3/M4**: 4-6x faster than CPU

Results depend on:
- Model architecture (CNNs benefit most)
- Batch size (larger = better GPU utilization)
- Memory bandwidth (unified memory helps)

### Comparison to NVIDIA GPUs

- Entry-level GPUs (GTX 1660, RTX 3050): Similar to M1/M2
- Mid-range GPUs (RTX 3070, RTX 4070): 2-3x faster than M3
- High-end GPUs (A100, H100): 10-50x faster than M3

Apple Silicon is excellent for learning and prototyping, but dedicated NVIDIA GPUs are faster for production training.

---

## Understanding MPS vs CUDA

### What is MPS?

**MPS (Metal Performance Shaders)** is Apple's GPU acceleration framework for machine learning on Apple Silicon.

- Uses Metal API (Apple's GPU programming framework)
- Optimized for unified memory architecture
- Supports PyTorch operations (similar to CUDA)

### MPS vs CUDA Differences

| Feature | CUDA (NVIDIA) | MPS (Apple) |
|---------|---------------|-------------|
| **Hardware** | NVIDIA GPUs | Apple Silicon |
| **API** | CUDA C/C++ | Metal Shading Language |
| **PyTorch Support** | Excellent | Good (improving) |
| **Custom Kernels** | Easy with PyTorch | Limited |
| **Ecosystem** | Mature (20+ years) | Growing (3+ years) |

### Can I run CUDA code on Mac?

**No.** CUDA is NVIDIA-specific and requires NVIDIA GPUs. However:
- You can **learn CUDA concepts** using this repository's documentation
- You can use Google Colab (free GPU) to run actual CUDA code
- MPS provides similar acceleration for PyTorch operations

---

## Understanding GPU Acceleration (Mental Model)

### For CUDA (when you move to NVIDIA GPUs):
- **CUDA** lets you run many lightweight **threads** organized into **blocks** (which form a **grid**) on the GPU
- **Warps** (groups of 32 threads) execute together; avoid divergent branching
- **Memory hierarchy**: prefer shared memory for data reused by a block; global memory is slower

### For MPS (Apple Silicon):
- **Threadgroups** (similar to CUDA blocks) execute in parallel
- **Threads** within a threadgroup can share memory
- **Unified memory** means CPU and GPU share the same RAM (no explicit copying)

Deep learning frameworks (PyTorch/TensorFlow) abstract these details and provide the same API across CUDA and MPS.

---

## Next Steps

1. ✅ Run the training experiments and compare CPU vs MPS performance
2. 📊 Explore the Jupyter notebooks in `notebooks/` directory
3. 🔬 Experiment with different batch sizes and model architectures
4. ☁️ Try Google Colab (free) to experience actual CUDA GPUs
5. 📚 Read about CUDA concepts (even though you can't run them on Mac)

---

## Learning CUDA from Mac

Even though you can't run CUDA on Mac, you can:

1. **Learn the concepts** from the documentation and code
2. **Use Google Colab** (free NVIDIA GPU access) for hands-on practice
3. **Read the CUDA kernel code** in `src/cuda/vec_add.cu` to understand the programming model
4. **Prototype with MPS** then deploy to cloud GPU instances

---

## Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Programming Guide](https://developer.apple.com/metal/)
- [Google Colab](https://colab.research.google.com/) - Free CUDA GPU access

**Happy GPU computing on Apple Silicon! 🚀**

