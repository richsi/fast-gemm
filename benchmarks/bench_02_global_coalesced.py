import torch
import triton
import fast_gemm

def benchmark_global_coalesced_layer():
  M = N = K = 4096
  print(f"Initializing standard benchmark boundary: {M} x {N} x {K}...")

  A = torch.randn((M, K), device='cuda', dtype=torch.float32)
  B = torch.randn((K, N), device='cuda', dtype=torch.float32)

  # profile torch
  ms_torch = triton.testing.do_bench(lambda: torch.matmul(A, B))
  tflops_torch = (2 * M * N * K) / (ms_torch * 1e-3) / 1e12

  # profile cuda
  ms_cuda = triton.testing.do_bench(lambda: fast_gemm.matmul_global_coalesced(A, B))
  tflops_cuda = (2 * M * N * K) / (ms_cuda * 1e-3) / 1e12

  print("\n" + "="*45)
  print(f"BENCHMARK RESULTS ({M}x{N}x{K})")
  print("="*45)
  print(f"PyTorch (cuBLAS) : {ms_torch:8.3f} ms | {tflops_torch:6.2f} TFLOP/s")
  print(f"Level 02 (Naive) : {ms_cuda:8.3f} ms | {tflops_cuda:6.2f} TFLOP/s")
  print("-"*45)
  print(f"Performance Gap  : Global coalesced is running at {((tflops_cuda / tflops_torch) * 100):.2f}% of cuBLAS speed.")
  print("="*45)

if __name__ == "__main__":
  benchmark_global_coalesced_layer()