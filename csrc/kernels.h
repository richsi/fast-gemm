#pragma once
#include <torch/extension.h>

torch::Tensor run_sgemm_naive(torch::Tensor A, torch::Tensor B);