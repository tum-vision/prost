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

#include "prost/prox/prox_ind_soc.hpp"
#include "prost/prox/vector.hpp"
#include "prost/prox/helper.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
__global__
void ProxIndSOCKernel(
  T *d_res,
  const T *d_arg,
  size_t count,
  size_t dim,
  T alpha)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> x(count, dim-1, false, tx, d_res);
    const Vector<const T> x0(count, dim-1, false, tx, d_arg);
    T& y = d_res[count * (dim-1) + tx];
    const T y0 = d_arg[count * (dim-1) + tx];

    T norm_x0 = 0;
    for(size_t i = 0; i < dim-1; i++) {
      norm_x0 += x0[i] * x0[i];
    }

    norm_x0 = sqrt(norm_x0);
    
    if(norm_x0 <= y0) {
      for(size_t i = 0; i < dim-1; i++) {
	x[i] = x0[i];
      }
      y = y0;	
    }
    else if(norm_x0 <= -y0) {
      for(size_t i = 0; i < dim-1; i++) {
	x[i] = 0;
      }
      y = 0;
    }
    else {
      T fac = (y0 + norm_x0) / (2 * norm_x0);

      for(size_t i = 0; i < dim-1; i++) {
	x[i] = fac * x0[i];
      }
      y = fac * norm_x0;
    }
  }
}


template<typename T>
void 
ProxIndSOC<T>::EvalLocal(
  const typename thrust::device_vector<T>::iterator& result_beg,
  const typename thrust::device_vector<T>::iterator& result_end,
  const typename thrust::device_vector<T>::const_iterator& arg_beg,
  const typename thrust::device_vector<T>::const_iterator& arg_end,
  const typename thrust::device_vector<T>::const_iterator& tau_beg,
  const typename thrust::device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  ProxIndSOCKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      this->count_,
      this->dim_,
      this->alpha_);
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

template<typename T>
void
ProxIndSOC<T>::Initialize() 
{
  if(alpha_ != 1) {
    throw Exception("ProxIndSOC: Only alpha = 1 implemented right now.");
  }
}

// Explicit template instantiation
template class ProxIndSOC<float>;
template class ProxIndSOC<double>;

}
