/*
 * This file is part of pdsolver.
 *
 * Copyright (C) 2015 Thomas Möllenhoff <thomas.moellenhoff@in.tum.de> 
 *
 * pdsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pdsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pdsolver. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "be_pdhg.h"

#include <cuComplex.h>
#include <iostream>
#include <sstream>
#include "../../utils/cuwrapper.h"

__global__
void pdhg_proxarg_primal(real *d_prox_arg, real *d_x, real tau,
                         real *d_precond_rgt, real *d_y_mat, int n)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx > n)
    return;

  d_prox_arg[idx] = d_x[idx] - tau * d_precond_rgt[idx] * d_y_mat[idx];
}

__global__
void pdhg_proxarg_dual(real *d_prox_arg, real *d_y, real sigma, real theta,
                       real *d_precond_lft, real *d_x_mat, real *d_x_mat_prev, int m)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx > m)
    return;

  d_prox_arg[idx] = d_y[idx] +
                    sigma * d_precond_lft[idx] *
                    ((1 + theta) * d_x_mat[idx] - theta * d_x_mat_prev[idx]);
}

__global__
void pdhg_res_primal(real *d_res_primal, real *d_x, real *d_x_prev, real *d_y_mat,
                     real *d_y_mat_prev, real tau, real *d_precond_rgt, int n)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx > n)
    return;

  d_res_primal[idx] = ((d_x_prev[idx] - d_x[idx]) / (tau * d_precond_rgt[idx])) -
                       (d_y_mat_prev[idx] - d_y_mat[idx]);
}

__global__
void pdhg_res_dual(real *d_res_dual, real *d_y, real *d_y_prev, real *d_x_mat,
                   real *d_x_mat_prev, real sigma, real *d_precond_lft,
                   real theta, int m)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx > m)
    return;

  // @todo: derive residual for theta!=1
  d_res_dual[idx] = ((d_y_prev[idx] - d_y[idx]) / (sigma * d_precond_lft[idx])) -
                     (d_x_mat_prev[idx] - d_x_mat[idx]);
}

bool be_pdhg::initialize() {
  m = mat->nrows();
  n = mat->ncols();
  l = std::max(m, n);

  // overall memory requirement is quite high -- can this be reduced?
  // around 5n + 6m + the sparse matrix... 
  cudaMalloc((void **)&d_x, n * sizeof(real));
  cudaMalloc((void **)&d_y, m * sizeof(real));
  cudaMalloc((void **)&d_prox_arg, l * sizeof(real));  
  cudaMalloc((void **)&d_x_prev, n * sizeof(real));
  cudaMalloc((void **)&d_y_prev, m * sizeof(real));
  cudaMalloc((void **)&d_x_mat, m * sizeof(real));
  cudaMalloc((void **)&d_x_mat_prev, m * sizeof(real));
  cudaMalloc((void **)&d_y_mat, n * sizeof(real));
  cudaMalloc((void **)&d_y_mat_prev, n * sizeof(real));
  cudaMalloc((void **)&d_res_primal, n * sizeof(real));
  cudaMalloc((void **)&d_res_dual, m * sizeof(real));

  tau = sigma = 1;
  theta = 1;

  // @todo initialize y_mat as K^T y^0.
  // @todo add custom initializations. for now, everything is zero
  cudaMemset(d_x, 0, n * sizeof(real));
  cudaMemset(d_x_prev, 0, n * sizeof(real));
  cudaMemset(d_y_mat, 0, n * sizeof(real));
  cudaMemset(d_y_mat_prev, 0, n * sizeof(real));
  cudaMemset(d_res_primal, 0, n * sizeof(real));
  cudaMemset(d_y, 0, m * sizeof(real));
  cudaMemset(d_y_prev, 0, m * sizeof(real));
  cudaMemset(d_res_dual, 0, m * sizeof(real));
  cudaMemset(d_x_mat, 0, m * sizeof(real));
  cudaMemset(d_x_mat_prev, 0, m * sizeof(real));
  cudaMemset(d_prox_arg, 0, l * sizeof(real));

  cublasCreate(&cublas_handle);
  
  return true;
}

void be_pdhg::do_iteration() {
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid_n((n + block.x - 1) / block.x, 1, 1);
  dim3 grid_m((m + block.x - 1) / block.x, 1, 1);

  // Compute d_prox_arg = x^k - tau * T * (K^T * y^k).
  pdhg_proxarg_primal<<<grid_n, block>>>(d_prox_arg,
                                         d_x,
                                         tau,
                                         d_precond_rgt,
                                         d_y_mat,
                                         n);
  cudaDeviceSynchronize();

  // d_x_prev = x^k
  std::swap(d_x, d_x_prev);

  // d_x = x^{k+1} = prox_g(d_prox_arg)
  for(int j = 0; j < prox_g.size(); ++j)
    prox_g[j]->eval(d_prox_arg, d_x, tau, d_precond_rgt);

  // d_x_mat_prev = K * x^k
  std::swap(d_x_mat, d_x_mat_prev);

  // d_x_mat = K * x^{k+1}
  mat->mv(d_x, d_x_mat, false, 1, 0);

  // d_prox_arg = d_y^k + sigma * S * K (x^{k+1} + theta (x^{k+1} - x^k))
  pdhg_proxarg_dual<<<grid_m, block>>>(d_prox_arg,
                                       d_y,
                                       sigma,
                                       theta,
                                       d_precond_lft,
                                       d_x_mat,
                                       d_x_mat_prev,
                                       m);
  cudaDeviceSynchronize();

  // d_y = y^{k+1} = prox_h(d_prox_arg)
  // d_y_prev = y^k
  std::swap(d_y, d_y_prev);
  for(int j = 0; j < prox_hc.size(); ++j)
    prox_hc[j]->eval(d_prox_arg, d_y, sigma, d_precond_lft);

  // d_y_mat = K^T * y^{k+1}
  // d_y_mat_prev = K^T * y^k
  std::swap(d_y_mat, d_y_mat_prev);
  mat->mv(d_y, d_y_mat, true, 1, 0);
  
  // compute residuals
  // res_primal = (x^k - x^{k+1}) / tT - K^T (y^k - y^{k+1})
  pdhg_res_primal<<<grid_n, block>>>(d_res_primal,
                                     d_x, d_x_prev,
                                     d_y_mat, d_y_mat_prev,
                                     tau, d_precond_rgt, n);

  // res_dual = (y^k - y^{k+1}) / sS - K (x^k - x^{k+1})
  pdhg_res_dual<<<grid_m, block>>>(d_res_dual,
                                   d_y, d_y_prev,
                                   d_x_mat, d_x_mat_prev,
                                   sigma, d_precond_lft, theta, m);
  cudaDeviceSynchronize();

  cuwrapper::asum<real>(cublas_handle, d_res_primal, n, &res_primal);
  cuwrapper::asum<real>(cublas_handle, d_res_dual, m, &res_dual);

  //std::cout << res_primal << "," << res_dual << std::endl;
  
  // adapt step-sizes according to chosen algorithm
  switch(opts.pdhg) {
    case kAlg1: // fixed step sizes
      break;

    case kAlg2: // adapt based on strong convexity gamma
      // @todo implement me
      break;

    case kAdapt: { // adapt based on residuals
      // @todo implement me
    } break;
  }
}

void be_pdhg::get_iterates(real *x, real *y) {
  cudaMemcpy(x, d_x, sizeof(real) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, sizeof(real) * m, cudaMemcpyDeviceToHost);
}

bool be_pdhg::is_converged() {
  //return std::max(res_primal, res_dual) < opts.tolerance;
  return false;
}

void be_pdhg::release() {
  cudaFree(d_prox_arg);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_x_prev);
  cudaFree(d_y_prev);
  cudaFree(d_x_mat);
  cudaFree(d_y_mat);
  cudaFree(d_x_mat_prev);
  cudaFree(d_y_mat_prev);
  cudaFree(d_res_primal);
  cudaFree(d_res_dual);

  cublasDestroy(cublas_handle);
}

std::string be_pdhg::status() {
  std::stringstream ss;

  ss << "Primal Residual: " << res_primal;
  ss << ", Dual Residual: " << res_dual << "\n"; 

  return ss.str();
}