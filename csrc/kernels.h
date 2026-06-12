#pragma once
#include <torch/extension.h>

torch::Tensor run_sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta);
torch::Tensor run_sgemm_global_coalesced(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta);
torch::Tensor run_sgemm_shared_mem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta);