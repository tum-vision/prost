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

#include "prox_sum_1d.h"

#include <cassert>
#include <iostream>
#include "prox_fun_1d.cuh"

template<class prox_fun_1d, typename real>
__global__
void prox_sum_1d_kernel(real *d_proxarg,
                        real *d_result,
                        real tau,
                        real *d_tau,
                        const prox_fun_1d prox,
                        const real *d_a,
                        const real *d_b,
                        const real *d_c,
                        const real *d_d,
                        const real *d_e,
                        real cf_a, real cf_b, real cf_c, real cf_d, real cf_e,
                        int idx, int cnt, bool invert_tau)
{
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx < cnt) {
    int global_idx = idx + th_idx;

    if(d_a != 0) cf_a = d_a[th_idx];
    if(d_b != 0) cf_b = d_b[th_idx];
    if(d_c != 0) cf_c = d_c[th_idx];
    if(d_d != 0) cf_d = d_d[th_idx];
    if(d_e != 0) cf_e = d_e[th_idx];

    real new_tau = tau;
    if(d_tau != 0)
      new_tau *= d_tau[th_idx];
    
    if(invert_tau)
      new_tau = real(1) / new_tau;

    real arg = cf_a * (d_proxarg[global_idx] * new_tau - cf_d) / (cf_e + new_tau) - cf_b;
    real step = (cf_e + new_tau) / (cf_c * cf_a * cf_a);
    
    d_result[global_idx] = (prox.apply(arg, step) + cf_b) / cf_a;
  }
}

prox_sum_1d::prox_sum_1d(int idx,
                         int cnt,
                         const prox_1d_coefficients& prox_coeffs,
                         const prox_fun_1d& fn)
    : prox(idx, cnt, 1, false, true), h_coeffs(prox_coeffs), prox_fn(fn)
{
  const std::vector<real>* coeffs[kNumCoeffs] = {
    &prox_coeffs.a,
    &prox_coeffs.b,
    &prox_coeffs.c,
    &prox_coeffs.d,
    &prox_coeffs.e };
  
  for(int i = 0; i < kNumCoeffs; i++)
  {
    assert(!(*coeffs[i]).empty());

    real *ptr = 0;
    if((*coeffs[i]).size() > 1)
    {
      assert((*coeffs[i]).size() == cnt);

      cudaMalloc((void **)&ptr, sizeof(real) * cnt);
      cudaMemcpy(ptr, &(*coeffs[i])[0], sizeof(real) * cnt, cudaMemcpyHostToDevice);
    }

    d_coeffs.push_back(ptr);
  }
}

prox_sum_1d::~prox_sum_1d() {
  for(int i = 0; i < kNumCoeffs; i++)
  {
    if(d_coeffs[i])
      cudaFree(d_coeffs[i]);
  }
}

void prox_sum_1d::eval(real *d_proxarg,
                       real *d_result,
                       real tau,
                       real *d_tau,
                       bool invert_tau) {
  
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count + block.x - 1) / block.x, 1, 1);

  const std::vector<real>* coeffs[kNumCoeffs] = {
    &h_coeffs.a,
    &h_coeffs.b,
    &h_coeffs.c,
    &h_coeffs.d,
    &h_coeffs.e };
  
  const real cf_a = (*coeffs[0])[0]; const real *d_a = d_coeffs[0];
  const real cf_b = (*coeffs[1])[0]; const real *d_b = d_coeffs[1];
  const real cf_c = (*coeffs[2])[0]; const real *d_c = d_coeffs[2];
  const real cf_d = (*coeffs[3])[0]; const real *d_d = d_coeffs[3];
  const real cf_e = (*coeffs[4])[0]; const real *d_e = d_coeffs[4];

  // @TODO this code could probably be automatically generated using macros
  // should make it more clear and faster to add new proxs. same in
  // prox_sum_norm2.cu
  switch(prox_fn)
  {   
    case kZero:
      prox_sum_1d_kernel<prox_1d_zero<real>, real>
          <<<grid, block>>>(d_proxarg,
                            d_result,
                            tau,
                            d_tau,
                            prox_1d_zero<real>(),
                            d_a, d_b, d_c, d_d, d_e,
                            cf_a, cf_b, cf_c, cf_d, cf_e,
                            index, count, invert_tau);
      break;

    case kAbs:
      prox_sum_1d_kernel<prox_1d_abs<real>, real>
          <<<grid, block>>>(d_proxarg,
                            d_result,
                            tau,
                            d_tau,
                            prox_1d_abs<real>(),
                            d_a, d_b, d_c, d_d, d_e,
                            cf_a, cf_b, cf_c, cf_d, cf_e,
                            index, count, invert_tau);
      break;

    case kIndicatorLeq:
      prox_sum_1d_kernel<prox_1d_leq<real>, real>
          <<<grid, block>>>(d_proxarg,
                            d_result,
                            tau,
                            d_tau,
                            prox_1d_leq<real>(),
                            d_a, d_b, d_c, d_d, d_e,
                            cf_a, cf_b, cf_c, cf_d, cf_e,
                            index, count, invert_tau);
      break;

    case kIndicatorEq:
      prox_sum_1d_kernel<prox_1d_eq<real>, real>
          <<<grid, block>>>(d_proxarg,
                            d_result,
                            tau,
                            d_tau,
                            prox_1d_eq<real>(),
                            d_a, d_b, d_c, d_d, d_e,
                            cf_a, cf_b, cf_c, cf_d, cf_e,
                            index, count, invert_tau);
      break;

    case kSquare:
      prox_sum_1d_kernel<prox_1d_square<real>, real>
          <<<grid, block>>>(d_proxarg,
                            d_result,
                            tau,
                            d_tau,
                            prox_1d_square<real>(),
                            d_a, d_b, d_c, d_d, d_e,
                            cf_a, cf_b, cf_c, cf_d, cf_e,
                            index, count, invert_tau);
      break;

    case kIndicatorAbsLeq:
      prox_sum_1d_kernel<prox_1d_absleq<real>, real>
          <<<grid, block>>>(d_proxarg,
                            d_result,
                            tau,
                            d_tau,
                            prox_1d_absleq<real>(),
                            d_a, d_b, d_c, d_d, d_e,
                            cf_a, cf_b, cf_c, cf_d, cf_e,
                            index, count, invert_tau);
      break;

    default:
      break;
  }

  cudaDeviceSynchronize();
}
