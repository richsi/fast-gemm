#include <torch/extension.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

__global__ void sgemm_shared_mem(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  float alpha, float beta,
  int M, int N, int K
) {

  const int thread_row = threadIdx.y;
  const int thread_col = threadIdx.x;

  const int row = blockIdx.y * BLOCK_SIZE + thread_row;
  const int col = blockIdx.x * BLOCK_SIZE + thread_col;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float val = 0.0f;

  // walk over K dimension with block size stride
  for (int t = 0; t < K; t += BLOCK_SIZE) {

    // each thread lodas ONE element of A and B
    int a_col = t + thread_col;
    int b_row = t + thread_row;

    As[thread_row][thread_col] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[thread_row][thread_col] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads(); // wait until entire tile is loaded

    // dot product over this tile
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      val += As[thread_row][k] * Bs[k][thread_col];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * val + beta * C[row * N + col];
  }
}

torch::Tensor run_sgemm_shared_mem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  const float* A_ptr = A.data_ptr<float>();
  const float* B_ptr = B.data_ptr<float>();
  float* C_ptr = C.data_ptr<float>();

  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

  sgemm_shared_mem<<<grid_dim, block_dim>>>(A_ptr, B_ptr, C_ptr, alpha, beta, M, N, K);

  return C;
}