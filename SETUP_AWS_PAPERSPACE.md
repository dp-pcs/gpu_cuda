# AWS / Paperspace Setup Guide

**GPU/CUDA Hands-On Tutorial for Cloud GPU Instances**

This guide covers setup for AWS EC2 GPU instances (g4dn, g5, p3, etc.) and Paperspace GPU machines.

---

## Prerequisites

### AWS
- AWS account with GPU instance access (g4dn.xlarge, g5.xlarge, or p3.2xlarge)
- NVIDIA GPU driver installed
- Docker with NVIDIA Container Toolkit installed

### Paperspace
- Paperspace account
- GPU machine (P4000, RTX4000, A4000, or better)
- Docker with NVIDIA Container Toolkit installed

---

## Quick Start (Recommended: Docker)

### 1. Clone the Repository

```bash
git clone git@github.com:dp-pcs/gpu_cuda.git
cd GPU_CUDA
```

### 2. Build the Docker Image

```bash
docker build -f env/Dockerfile.cuda -t cuda-tutorial .
```

### 3. Run the Container

```bash
docker run --gpus all -it -p 8888:8888 -v $PWD:/workspace cuda-tutorial bash
```

The `--gpus all` flag gives the container access to all GPUs on the host.

### 4. Verify GPU Access (inside container)

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

### 5. Run Training Experiments

```bash
# CPU baseline
python src/train.py --device cpu --epochs 1 --batch-size 128

# GPU accelerated
python src/train.py --device cuda --epochs 1 --batch-size 128

# Profile GPU usage
python src/profiler.py --device cuda
```

---

## Alternative: Native Installation (without Docker)

### 1. Install NVIDIA Drivers and CUDA

**AWS EC2 (Ubuntu/Debian)**
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Install CUDA toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

**Paperspace** typically comes with drivers pre-installed.

### 2. Install Python Dependencies

```bash
# Clone repository
git clone git@github.com:dp-pcs/gpu_cuda.git
cd GPU_CUDA

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib onnx onnxruntime
```

### 3. Verify Setup

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Running Experiments

### Basic Training

```bash
# Activate environment (if not using Docker)
source .venv/bin/activate

# Run CPU vs CUDA comparison
python src/train.py --device cpu --epochs 1 --batch-size 128
python src/train.py --device cuda --epochs 1 --batch-size 256

# Profile GPU utilization
python src/profiler.py --device cuda
```

### Monitor GPU Usage

Open two terminal sessions:

**Terminal 1** - Monitor GPU:
```bash
watch -n 1 nvidia-smi
```

**Terminal 2** - Run training:
```bash
python src/train.py --device cuda --epochs 5 --batch-size 256
```

### Build and Run Custom CUDA Kernel

```bash
cd src/cuda
./build.sh
./vec_add
```

### ONNX Export and TensorRT Inference

```bash
# Export model to ONNX
python src/export_onnx.py

# Run TensorRT benchmark
python src/infer_trt.py
```

---

## Using Makefile Commands

```bash
make verify          # Verify GPU setup
make train-cpu       # Train on CPU
make train-cuda      # Train on CUDA
make export-onnx     # Export to ONNX
make trt-benchmark   # TensorRT inference benchmark
```

---

## Jupyter Notebook (Optional)

### From Docker Container

```bash
# Inside container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access from your browser at `http://<instance-ip>:8888`

### Native Installation

```bash
source .venv/bin/activate
pip install jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

---

## Troubleshooting

### NVIDIA Container Toolkit not installed (Docker)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### CUDA version mismatch

Check your CUDA version:
```bash
nvcc --version
nvidia-smi  # Look for "CUDA Version" in top right
```

Install matching PyTorch wheels:
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`

### Out of Memory (OOM)

- Lower `--batch-size`: try 128, 64, 32, or 16
- Use smaller models for large batch experiments
- Monitor GPU memory: `nvidia-smi` shows memory usage

### TensorRT Issues

- Ensure TensorRT is installed (included in Docker image)
- Check CUDA and cuDNN compatibility with TensorRT version
- You can skip TensorRT steps and focus on PyTorch CUDA

---

## Understanding CUDA (Mental Model)

- **CUDA** lets you run many lightweight **threads** organized into **blocks** (which form a **grid**) on the GPU
- **Warps** (groups of 32 threads) execute together; avoid divergent branching for best performance
- **Memory hierarchy**: prefer shared memory for data reused by a block; global memory is slower
- Deep learning frameworks (PyTorch/TensorFlow) use CUDA/cuDNN to automatically accelerate operations

---

## Cost Management Tips

### AWS
- Use **Spot Instances** for up to 70% savings
- Stop instances when not in use (EBS volumes persist)
- Use `g4dn.xlarge` for learning (cheaper than p3 instances)

### Paperspace
- Use **hourly billing** for short experiments
- Stop machines when done (storage billed separately)
- Use **P4000** or **RTX4000** for cost-effective learning

---

## Next Steps

1. Explore the Jupyter notebooks in `notebooks/` directory
2. Experiment with different batch sizes and model architectures
3. Profile and optimize your training code
4. Try custom CUDA kernels in `src/cuda/`

**Happy GPU computing! 🚀**

