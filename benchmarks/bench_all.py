import torch
import triton
import fast_gemm

def run_suite():
  M = N = K = 4096
  print(f"Initializing standard benchmark boundary: {M} x {N} x {K}...\n")

  A = torch.randn((M, K), device='cuda', dtype=torch.float32)
  B = torch.randn((K, N), device='cuda', dtype=torch.float32)

  total_flops = 2 * M * N * K

  kernels = [
    ("PyTorch (cuBLAS)", lambda: torch.matmul(A, B)),
    ("Level 01: Naive", lambda: fast_gemm.matmul_naive(A, B)),
    ("Level 02: Global Coalesced", lambda: fast_gemm.matmul_global_coalesced(A, B)),
  ]

  print(f"{'Kernel Implementation':<28} | {'Latency (ms)':<14} | {'TFLOP/s':<10} | {'% of cuBLAS':<12}")
  print("-" * 64)

  baseline_tflops = None

  for name, func in kernels:
    # latency
    ms = triton.testing.do_bench(func)
        
    # throughput
    tflops = (total_flops / (ms * 1e-3)) / 1e12
        
    if name == "PyTorch (cuBLAS)":
      baseline_tflops = tflops
      percent_cublas = 100.0
    else:
      percent_cublas = (tflops / baseline_tflops) * 100

    # Print formatted row
    print(f"{name:<28} | {ms:14.3f} | {tflops:10.2f} | {percent_cublas:10.2f}%")

  print("-" * 73)

if __name__ == "__main__":
  run_suite()