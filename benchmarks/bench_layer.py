import torch
import triton
import fast_gemm
import argparse

def benchmark_layer(kernel_type="naive", M=4096, N=4096, K=4096):
  """
  Benchmarks custom CUDA matmul kernels against PyTorch (cuBLAS).
    
  Args:
    kernel_type (str): The kernel to benchmark. Options: 'naive', 'coalesced'.
    M, N, K (int): Matrix dimensions.
  """
  print(f"Initializing standard benchmark boundary: {M} x {N} x {K}...")
  print(f"Target Kernel: {kernel_type}")

  A = torch.randn((M, K), device='cuda', dtype=torch.float32)
  B = torch.randn((K, N), device='cuda', dtype=torch.float32)

  # PyTorch (cuBLAS baseline)
  ms_torch = triton.testing.do_bench(lambda: torch.matmul(A, B))
  tflops_torch = (2 * M * N * K) / (ms_torch * 1e-3) / 1e12

  if kernel_type == "naive":
    target_kernel = fast_gemm.matmul_naive
    display_name = "Level 01 (Naive)"
  elif kernel_type == "coalesced":
    target_kernel = fast_gemm.matmul_global_coalesced
    display_name = "Level 02 (Coalesced)"
  elif kernel_type == "smem":
    target_kernel = fast_gemm.matmul_shared_mem
    display_name = "Level 03 (Shared Memory)"
  else:
    raise ValueError("Invalid kernel_type.")

  # profile the selected kernel
  ms_cuda = triton.testing.do_bench(lambda: target_kernel(A, B))
  tflops_cuda = (2 * M * N * K) / (ms_cuda * 1e-3) / 1e12

  print("\n" + "="*50)
  print(f"BENCHMARK RESULTS ({M}x{N}x{K})")
  print("="*50)
  print(f"PyTorch (cuBLAS)   : {ms_torch:8.3f} ms | {tflops_torch:6.2f} TFLOP/s")
  print(f"{display_name:<18} : {ms_cuda:8.3f} ms | {tflops_cuda:6.2f} TFLOP/s")
  print("-" * 50)
  print(f"Performance Gap    : Kernel is running at {((tflops_cuda / tflops_torch) * 100):.2f}% of cuBLAS speed.")
  print("="*50)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-k", "--kernel", help="Select which kernel to run.")

  args = parser.parse_args()
    
  benchmark_layer(kernel_type=args.kernel)