import torch
import fast_gemm_cuda

def matmul_naive(A, B, C=None, alpha=1.0, beta=0.0):
  if C is None:
    C = torch.zeros((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

  return fast_gemm_cuda.matmul_naive(A, B, C, alpha, beta)


def matmul_global_coalesced(A, B, C=None, alpha=1.0, beta=0.0):
  if C is None:
    C = torch.zeros((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

  return fast_gemm_cuda.matmul_global_coalesced(A, B, C, alpha, beta)