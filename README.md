# fast-gemm

A personal learning project: writing SGEMM (FP32 matrix multiply) CUDA kernels from scratch and benchmarking them against cuBLAS. Each kernel is a "level" applying a new optimization, loosely following the classic [how-to-optimize-gemm](https://siboehm.com/articles/22/CUDA-MMM) progression.

Computes `C = alpha * (A @ B) + beta * C`.

## Kernels

Kernel Implementation        | Latency (ms)   | TFLOP/s    | % of cuBLAS 
----------------------------------------------------------------
PyTorch (cuBLAS)             |          2.089 |      65.79 |     100.00%
Level 01: Naive              |        143.744 |       0.96 |       1.45%
Level 02: Global Coalesced   |         19.608 |       7.01 |      10.65%
Level 03: Shared Memory      |         15.023 |       9.15 |      13.91%
-------------------------------------------------------------------------

## Setup

Built and tested on:
- RTX 5090
- Ubuntu 24.04
- CUDA 13.0

Build the CUDA extension in place:

```bash
pip install torch triton
pip install -e . --no-build-isolation
```

## Usage

```python
import torch
import fast_gemm

A = torch.randn(4096, 4096, device="cuda")
B = torch.randn(4096, 4096, device="cuda")

C = fast_gemm.matmul_global_coalesced(A, B)
```

## Benchmarks

```bash
# run all kernels vs cuBLAS
python benchmarks/bench_all.py

# run a single kernel: naive | coalesced
python benchmarks/bench_layer.py -k coalesced
```
