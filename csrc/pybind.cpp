#include <torch/extension.h>
#include "kernels.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  return run_sgemm_naive(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_naive", &matmul_naive, "Level 01 Naive GEMM");
}