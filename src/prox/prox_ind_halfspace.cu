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

#include "prost/prox/prox_ind_halfspace.hpp"
#include "prost/prox/vector.hpp"
#include "prost/prox/helper.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

// project the vector v onto the halfspace described by the set
// { x | <n,x> <= t }
template<typename T>
inline __host__ __device__
void ProjectHalfspace(Vector<const T> const& v,
		      Vector<const T> const& n,
		      T t,
		      Vector<T>& result,
		      int dim)
{
  T sq_norm = 0;
  T iprod = 0;
  for(size_t i = 0; i < dim; i++) {
    sq_norm += n[i] * n[i];
    iprod += n[i] * v[i];
  }

  for(size_t i = 0; i < dim; i++) {
    result[i] = v[i] - (max(static_cast<T>(0), iprod - t) / sq_norm) * n[i];
  }  
}
  
template<typename T>
__global__
void ProxIndHalfspaceKernel(
  T *d_res,
  const T *d_arg,
  size_t count,
  size_t dim,
  const T *d_a,
  const T *d_b,
  size_t sz_a,
  size_t sz_b)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> x(count, dim, false, tx, d_res);
    const Vector<const T> x0(count, dim, false, tx, d_arg);

    T b;
    if(sz_b == count) 
      b = d_b[tx];
    else 
      b = d_b[0];

    if(sz_a == count * dim) {
      const Vector<const T> a(count, dim, false, tx, d_a);

      ProjectHalfspace<T>(x0, a, b, x, dim);
    }
    else {
      const Vector<const T> a(count, dim, true, 0, d_a);

      ProjectHalfspace<T>(x0, a, b, x, dim);
    }
  }
}


template<typename T>
void 
ProxIndHalfspace<T>::EvalLocal(
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

  ProxIndHalfspaceKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      this->count_,
      this->dim_,
      thrust::raw_pointer_cast(&d_a_[0]),
      thrust::raw_pointer_cast(&d_b_[0]),
      a_.size(),
      b_.size());
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
ProxIndHalfspace<T>::Initialize() 
{
    if(a_.size() != this->count_ * this->dim_ && a_.size() != this->dim_)
      throw Exception("Wrong input: Coefficient a has to have dimension count*dim or dim!");

    if(b_.size() != this->count_ && b_.size() != 1)
      throw Exception("Wrong input: Coefficient b has to have dimension count or 1!");
    
    try
    {
      d_a_ = a_;
      d_b_ = b_;
    }
    catch(std::bad_alloc &e)
    {
      throw Exception(e.what());
    }
    catch(thrust::system_error &e)
    {
      throw Exception(e.what());
    }
    
}

// Explicit template instantiation
template class ProxIndHalfspace<float>;
template class ProxIndHalfspace<double>;

}
