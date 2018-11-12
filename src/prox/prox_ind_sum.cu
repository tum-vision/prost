/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <sstream>

#include "prost/prox/prox_ind_sum.hpp"
#include "prost/prox/vector.hpp"
#include "prost/prox/helper.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
__global__
void ProxIndSumKernel(
  T *d_res,
  const T *d_arg,
  const T *d_tau,
  const size_t *d_inds,
  size_t count,
  size_t dim,
  T tau,
  bool inv_tau)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {

    T sum_arg = 0;
    T sum_tau = 0;
    
    for(int i = 0; i < dim; i++) {
      T mytau = d_tau[d_inds[tx * dim + i]] * tau;
      if(inv_tau) mytau = 1. / mytau;
      
      sum_arg += d_arg[d_inds[tx * dim + i]];
      sum_tau += mytau;
    }

    for(int i = 0; i < dim; i++) {
      T mytau = d_tau[d_inds[tx * dim + i]] * tau;
      if(inv_tau) mytau = 1. / mytau;

      d_res[d_inds[tx * dim + i]] =
        d_arg[d_inds[tx * dim + i]] - mytau * (sum_arg - 1.) / sum_tau;
      
    }
    
  }
}

template<typename T>
void ProxIndSum<T>::Initialize() {

  if(count_ * dim_ != inds_.size())
    throw Exception("ProxIndSum: dimensions dont fit");
  
  d_inds_.resize(inds_.size());
  d_inds_ = inds_;
}

template<typename T>
void ProxIndSum<T>::Release() {
}

template<typename T>
size_t ProxIndSum<T>::gpu_mem_amount() const {
  return inds_.size() * sizeof(size_t);
}

template<typename T>
void ProxIndSum<T>::EvalLocal(
  const typename device_vector<T>::iterator& result_beg,
  const typename device_vector<T>::iterator& result_end,
  const typename device_vector<T>::const_iterator& arg_beg,
  const typename device_vector<T>::const_iterator& arg_end,
  const typename device_vector<T>::const_iterator& tau_beg,
  const typename device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count_ + block.x - 1) / block.x, 1, 1);

  // zero prox on other indices
  thrust::copy(arg_beg, arg_end, result_beg);
  cudaDeviceSynchronize();
  
  ProxIndSumKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      thrust::raw_pointer_cast(&d_inds_[0]),
      count_,
      dim_,
      tau,
      invert_tau);
  
  cudaDeviceSynchronize();
}

// Explicit template instantiation
template class ProxIndSum<float>;
template class ProxIndSum<double>;

}