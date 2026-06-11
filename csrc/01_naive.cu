#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sgemm_naive_kernel(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int M, int N, int K
) {
  // compute global row and col indices for this thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float val = 0.0f;

    // accumulate over dimension K
    for (int i = 0; i < K; ++i) {
      val += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = val;
  }
}


torch::Tensor run_sgemm_naive(torch::Tensor A, torch::Tensor B) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  // allocate space
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  // define thread config
  dim3 block_dim(32, 32);
  // ceiling alternative
  dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

  // extract continuous raw float memory references
  const float* A_ptr = A.data_ptr<float>();
  const float* B_ptr = B.data_ptr<float>();
  float* C_ptr = C.data_ptr<float>();

  sgemm_naive_kernel<<<grid_dim, block_dim>>>(A_ptr, B_ptr, C_ptr, M, N, K);

  return C;
}