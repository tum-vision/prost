#include "solver/solver_backend_pdhgtiny.hpp"

#include <iostream>
#include <sstream>

/**
 * @brief ...
 */
template<typename T>
__global__
void ComputeProxArgPrimalPDHGTiny(
    T *d_prox_arg,
    T *d_x,
    T tau,
    T *d_right,
    T *d_kty,
    size_t n) 
{  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx >= n)
    return;

  d_prox_arg[idx] = d_x[idx] - tau * d_right[idx] * d_kty[idx];
}

/**
 * @brief ...
 */
template<typename T>
__global__
void ComputeOverrelaxationPDHGTiny(
    T *d_x_bar,
    T *d_x,
    size_t n) 
{  
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx >= n)
    return;

  d_x_bar[idx] = static_cast<T>(2) * d_x[idx] - d_x_bar[idx];
}

/**
 * @brief ...
 */
template<typename T>
__global__
void ComputeProxArgDualPDHGTiny(
    T *d_prox_arg,
    T *d_y,
    T sigma,
    T *d_left,
    size_t m) 
{  
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx >= m)
    return;

  d_prox_arg[idx] = d_y[idx] + sigma * d_left[idx] * d_prox_arg[idx];
}

void SolverBackendPDHGTiny::PerformIteration() {
  size_t n = problem_.linop->ncols();
  size_t m = problem_.linop->nrows();
  
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid_n((n + block.x - 1) / block.x, 1, 1);
  dim3 grid_m((m + block.x - 1) / block.x, 1, 1);

  // d_temp_ <- proxarg(d_x, d_temp, ...)
  ComputeProxArgPrimalPDHGTiny<real>
      <<<grid_n, block>>>(d_temp_,
                          d_x_,
                          tau_,
                          problem_.precond->right(),
                          d_temp_,
                          n);

  std::swap(d_x_, d_x_bar_);

  // d_x <- prox(d_temp)
  for(size_t j = 0; j < problem_.prox_g.size(); ++j)
    problem_.prox_g[j]->Eval(
        d_temp_,
        d_x_,
        problem_.precond->right(),
        tau_);

  // d_x_bar <- 2d_x - d_x_bar
  ComputeOverrelaxationPDHGTiny<real>
      <<<grid_n, block>>>(d_x_bar_,
                          d_x_,
                          n);

  problem_.linop->Eval(d_temp_, d_x_bar_);

  // d_temp <- proxarg(d_y, d_temp, ...)
  ComputeProxArgDualPDHGTiny<real>
      <<<grid_m, block>>>(d_temp_,
                          d_y_,
                          sigma_,
                          problem_.precond->left(),
                          m);

  // d_y <- prox(d_temp)
  for(size_t j = 0; j < problem_.prox_hc.size(); ++j)
    problem_.prox_hc[j]->Eval(
        d_temp_,
        d_y_,
        problem_.precond->left(),
        sigma_);

  // d_temp <- KT d_y
  problem_.linop->EvalAdjoint(d_temp_, d_y_);
}

bool SolverBackendPDHGTiny::Initialize() {
  size_t m = problem_.linop->nrows();
  size_t n = problem_.linop->ncols();
  size_t l = std::max(m, n);

  cudaMalloc((void **)&d_x_, n * sizeof(real)); 
  cudaMalloc((void **)&d_x_bar_, n * sizeof(real)); 
  cudaMalloc((void **)&d_y_, m * sizeof(real)); 
  cudaMalloc((void **)&d_temp_, l * sizeof(real)); 

  cudaMemset(d_x_, 0, n * sizeof(real)); 
  cudaMemset(d_x_bar_, 0, n * sizeof(real)); 
  cudaMemset(d_y_, 0, m * sizeof(real)); 
  cudaMemset(d_temp_, 0, l * sizeof(real)); 

  tau_ = opts_.tau0;
  sigma_ = opts_.sigma0;

  return true;
}

void SolverBackendPDHGTiny::Release() {
  cudaFree(d_x_);
  cudaFree(d_x_bar_);
  cudaFree(d_y_);
  cudaFree(d_temp_);
}

void SolverBackendPDHGTiny::iterates(real *primal, real *dual) {
  cudaMemcpy(primal, d_x_, sizeof(real) * problem_.linop->ncols(), cudaMemcpyDeviceToHost);
  cudaMemcpy(dual, d_y_, sizeof(real) * problem_.linop->nrows(), cudaMemcpyDeviceToHost);
}

bool SolverBackendPDHGTiny::converged() {
  return false;
}

std::string SolverBackendPDHGTiny::status() {
  return std::string("");
}

size_t SolverBackendPDHGTiny::gpu_mem_amount() {
  int m = problem_.linop->nrows();
  int n = problem_.linop->ncols();

  return (2 * n + m + std::max(n, m)) * sizeof(real);
}
