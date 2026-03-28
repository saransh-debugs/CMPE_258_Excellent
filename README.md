# Homework 1 — Excellent: CUDA Core vs Tensor Core GEMM Benchmark

## Overview

This project measures the performance difference between traditional CUDA core FP32 matrix multiplication and TF32 Tensor Core matrix multiplication using a fully connected layer. The benchmark is implemented in both PyTorch and CUDA C++ (cuBLAS), and all experiments were run on an NVIDIA A100 GPU (High RAM) via Google Colab.

## Repository Structure

```
.
├── README.md
├── hw1_excellent_gemm_benchmark.ipynb   # Complete Colab notebook (PyTorch + cuBLAS + plots)
└── gemm_benchmark_results.png           # Output plot
```

## How to Run

1. Open `hw1_excellent_gemm_benchmark.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type** and select **A100 GPU** with High RAM enabled. TF32 requires Ampere architecture or newer.
3. Click **Runtime → Run all**. The notebook handles everything: GPU verification, PyTorch benchmark, CUDA C++ compilation and execution, result parsing, and plot generation.

No additional dependencies are needed. PyTorch, CUDA, and nvcc come preinstalled on Colab GPU runtimes.

## Benchmark Design

**PyTorch path:**
- Mode 1 (CUDA Core baseline): `torch.backends.cuda.matmul.allow_tf32 = False`
- Mode 2 (Tensor Core): `torch.backends.cuda.matmul.allow_tf32 = True`
- Uses `nn.Linear` for the FC layer with FP32 weights and input.

**cuBLAS path:**
- Baseline: `cublasSgemm` with `CUBLAS_DEFAULT_MATH`
- Tensor Core: `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_TF32`

**Timing methodology:**
- 20 warm-up iterations before each timed run
- 100 timed iterations per configuration
- GPU-side timing via CUDA events (`cudaEventRecord` / `torch.cuda.Event`)
- Explicit synchronization before reading elapsed time
- Throughput computed as 2MNK / avg_latency

**Matrix sizes tested:** 256, 512, 1024, 2048, 4096, 8192 (square, M=N=K)

## Results

The TF32 Tensor Core path delivers increasing speedup as matrix size grows. At size 256 the benefit is negligible due to insufficient work to saturate the Tensor Cores. At size 8192 the speedup exceeds 6.5x in both PyTorch and cuBLAS, with TF32 throughput climbing past 120 TFLOPS while the FP32 CUDA core path plateaus around 20 TFLOPS.

PyTorch and cuBLAS results are consistent with each other, which is expected since PyTorch delegates its matrix multiplications to cuBLAS internally.

The numerical accuracy cost of TF32 is small. At size 2048, mean relative error between FP32 and TF32 outputs was on the order of 1e-4, with a maximum around 1e-3. This is well within acceptable bounds for deep learning workloads.

## Hardware

- **GPU:** NVIDIA A100 (Ampere, compute capability 8.0), High RAM configuration
- **Environment:** Google Colab
