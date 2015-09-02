#include "prox_moreau.hpp"

__global__
void MoreauPrescale(
    real *d_arg,
    real tau,
    real *d_tau,
    int index,
    int total_count) {
  
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx >= total_count)
    return;

  int global_idx = index + th_idx;
  d_arg[global_idx] = d_arg[global_idx] / (tau * d_tau[global_idx]);
}

__global__
void MoreauPostscale(
    real *d_result,
    real *d_arg,
    real tau,
    real *d_tau,
    int index,
    int total_count) {
  
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx >= total_count)
    return;

  int global_idx = index + th_idx;
  d_result[global_idx] = tau * d_tau[global_idx] * (d_arg[global_idx] - d_result[global_idx]);
}

ProxMoreau::ProxMoreau(Prox *conjugate)
    : Prox(*conjugate), conjugate_(conjugate) {
}

ProxMoreau::~ProxMoreau() {
}

void ProxMoreau::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step) {

  int total_count = count_ * dim_;
  
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((total_count + block.x - 1) / block.x, 1, 1);

  MoreauPrescale<<<grid, block>>>(d_arg, tau, d_tau, index_, total_count);
  cudaDeviceSynchronize();
  
  conjugate_->Evaluate(d_arg, d_result, tau, d_tau, true);

  MoreauPostscale<<<grid, block>>>(d_result, d_arg, tau, d_tau, index_, total_count);
  cudaDeviceSynchronize();
}
