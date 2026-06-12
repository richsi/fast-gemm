#include <torch/extension.h>
#include "kernels.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  return run_sgemm_naive(A, B, C, alpha, beta);
}

torch::Tensor matmul_global_coalesced(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  return run_sgemm_global_coalesced(A, B, C, alpha, beta);
}

torch::Tensor matmul_shared_mem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  return run_sgemm_shared_mem(A, B, C, alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 01
  m.def("matmul_naive", &matmul_naive, "Level 01 Naive GEMM");
  // 02
  m.def("matmul_global_coalesced", &matmul_global_coalesced, "Level 02 Global Coalesced GEMM");
  // 03
  m.def("matmul_shared_mem", &matmul_shared_mem, "Level 03 Shared Memory");
}