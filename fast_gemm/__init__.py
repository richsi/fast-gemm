import torch
import functools
import fast_gemm_cuda

def auto_allocate_c(cuda_func):
  """Decorator to auto allocate output tensor C if not provided."""
  @functools.wrap(cuda_func)
  def wrapper(A, B, C=None, alpha=1.0, beta=0.0):
    if C is None:
      C = torch.zeros((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
    return cuda_func(A, B, C, alpha, beta)
  return wrapper

# expose functions
matmul_naive = auto_allocate_c(fast_gemm_cuda.matmul_naive)
matmul_global_coalesced = auto_allocate_c(fast_gemm_cuda.matmul_global_coalesced)