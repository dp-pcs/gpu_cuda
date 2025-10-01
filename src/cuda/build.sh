#!/usr/bin/env bash
set -euo pipefail
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc (CUDA toolkit) not found. Install CUDA or use the Dockerfile.cuda image."; exit 1
fi
nvcc -O2 -o vec_add vec_add.cu
echo "Built vec_add. Run ./vec_add"
