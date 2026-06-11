import fast_gemm_cuda

def matmul_naive(A, B):
  return fast_gemm_cuda.matmul_naive(A, B)