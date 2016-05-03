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

#include "prost/prox/shared_mem.hpp"
#include "prost/prox/vector.hpp"

#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T, class ELEM_OPERATION>
__global__
void ProxElemOperationKernel(
  T *d_res,
  const T *d_arg,
  const T *d_tau,
  T tau,
  bool invert_tau,
  size_t count,
  size_t dim,
  bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) 
  {
    Vector<T> res(count, dim, interleaved, tx, d_res);
    const Vector<const T> arg(count, dim, interleaved, tx, d_arg);
    const Vector<const T> tau_diag(count, dim, interleaved, tx, d_tau);

    SharedMem<typename ELEM_OPERATION::SharedMemType, typename ELEM_OPERATION::GetSharedMemCount> sh_mem(dim, threadIdx.x);

    ELEM_OPERATION op(dim, sh_mem);
    op(res, arg, tau_diag, tau, invert_tau);
  }
}

template<typename T, class ELEM_OPERATION>
__global__
void ProxElemOperationKernel(
  T *d_res,
  const T *d_arg,
  const T *d_tau,
  T tau,
  bool invert_tau,
  size_t count,
  size_t dim,
  ElemOpCoefficients<T, ELEM_OPERATION> coeffs,
  bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> res(count, dim, interleaved, tx, d_res);
    const Vector<const T> arg(count, dim, interleaved, tx, d_arg);
    const Vector<const T> tau_diag(count, dim, interleaved, tx, d_tau);

    SharedMem<typename ELEM_OPERATION::SharedMemType, typename ELEM_OPERATION::GetSharedMemCount> sh_mem(dim, threadIdx.x);

    T coeffs_local[ELEM_OPERATION::kCoeffsCount];
    for(int i = 0; i < ELEM_OPERATION::kCoeffsCount; i++)
    {
      if(coeffs.dev_p[i] == nullptr) 
        coeffs_local[i] = coeffs.val[i];
      else 
        coeffs_local[i] = coeffs.dev_p[i][tx];
    }

    ELEM_OPERATION op(coeffs_local, dim, sh_mem);
    op(res, arg, tau_diag, tau, invert_tau);
  }
}

template<typename T, class ELEM_OPERATION>
void 
ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::kCoeffsCount == 0>::type>::EvalLocal(
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

  typename ELEM_OPERATION::GetSharedMemCount get_shared_mem_count;

  size_t shmem_bytes =
    get_shared_mem_count(this->dim_) *
    block.x *
    sizeof(typename ELEM_OPERATION::SharedMemType);
  
  ProxElemOperationKernel<T, ELEM_OPERATION>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      tau,
      invert_tau,
      this->count_,
      this->dim_,
      this->interleaved_);
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

template<typename T, class ELEM_OPERATION>
void 
ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::kCoeffsCount != 0>::type>::EvalLocal(
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

  ElemOpCoefficients<T, ELEM_OPERATION> coeffs;
     
  for(size_t i = 0; i < ELEM_OPERATION::kCoeffsCount; i++)
  {
    if(coeffs_[i].size() > 1) 
      coeffs.dev_p[i] = thrust::raw_pointer_cast(&d_coeffs_[i][0]);
    else
    {
      coeffs.dev_p[i] = nullptr;
      coeffs.val[i] = coeffs_[i][0];
    }
  }

  typename ELEM_OPERATION::GetSharedMemCount get_shared_mem_count;

  size_t shmem_bytes =
    get_shared_mem_count(this->dim_) *
    block.x *
    sizeof(typename ELEM_OPERATION::SharedMemType);
  
  ProxElemOperationKernel<T, ELEM_OPERATION>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      tau,
      invert_tau,
      this->count_,
      this->dim_,
      coeffs,
      this->interleaved_);
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

template<typename T, class ELEM_OPERATION>
void
ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::kCoeffsCount != 0>::type>::Initialize() 
{
  for(size_t i = 0; i < ELEM_OPERATION::kCoeffsCount; i++)
  { 
    try
    {
      if(coeffs_[i].size() > 1)
      {
        d_coeffs_[i] = coeffs_[i];
      }
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
}

} // namespace prost
