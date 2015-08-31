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

#include "prox_moreau.h"

__global__
void moreau_prescale(real *d_proxarg, real tau, real *d_tau, int cnt)
{
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx >= cnt)
    return;

  d_proxarg[th_idx] = d_proxarg[th_idx] / (tau * d_tau[th_idx]);
}

__global__
void moreau_postscale(real *d_result, real *d_proxarg, real tau, real *d_tau, int cnt)
{
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx >= cnt)
    return;

  d_result[th_idx] = tau * d_tau[th_idx] * (d_proxarg[th_idx] - d_result[th_idx]);
}

prox_moreau::prox_moreau(prox *child)
  : child_prox(child)
{
  index = child->get_index();
  count = child->get_count();
  dim = child->get_dim();
  interleaved = child->is_interleaved();
  diagprox = child->is_diagprox();
}

prox_moreau::~prox_moreau() {
}

void prox_moreau::eval(real *d_proxarg,
                       real *d_result,
                       real tau,
                       real *d_tau,
                       bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count + block.x - 1) / block.x, 1, 1);

  moreau_prescale<<<grid, block>>>(d_proxarg, tau, d_tau, count);
  cudaDeviceSynchronize();
  
  child_prox->eval(d_proxarg, d_result, tau, d_tau, true);

  moreau_postscale<<<grid, block>>>(d_result, d_proxarg, tau, d_tau, count);
  cudaDeviceSynchronize();
}
