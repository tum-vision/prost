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
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include "prost/prox/prox_permute.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
__global__
void ProxPermuteKernel(
  T *d_res,
  const T *d_arg,
  const int *d_perm,
  size_t count,
  bool inverse)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    if(inverse)
      d_res[d_perm[tx]] = d_arg[tx];
    else
      d_res[tx] = d_arg[d_perm[tx]];
  }
}
  
template<typename T>
ProxPermute<T>::ProxPermute(std::shared_ptr<Prox<T> > base_prox, const std::vector<int>& perm)
  : Prox<T>(*base_prox), base_prox_(base_prox), perm_host_(perm)
{
}

template<typename T>
ProxPermute<T>::~ProxPermute() 
{
}

template<typename T>
void ProxPermute<T>::Initialize() 
{
  for(int i = 0; i < perm_host_.size(); i++) {
    std::cout << perm_host_[i] << " ";
  }
  std::cout << std::endl;
  
  try 
  {
    permuted_arg_.resize(this->size_);
    perm_ = perm_host_;
  } 
  catch(const std::bad_alloc &e)
  {
    std::stringstream ss;
    ss << "Out of memory: " << e.what();

    throw Exception(ss.str());
  }

  if(perm_host_.size() != base_prox_->size()) {
    std::stringstream ss;
    ss << "Permutation vector has wrong size (" << perm_host_.size() << ") instead of " << base_prox_->size() << ".";
      
    throw Exception(ss.str());
  }


  base_prox_->Initialize();
}

template<typename T>
void ProxPermute<T>::Release() 
{
  base_prox_->Release();
}

template<typename T>
void ProxPermute<T>::EvalLocal(
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
  dim3 grid((perm_host_.size() + block.x - 1) / block.x, 1, 1);

  // permute argument
  ProxPermuteKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&perm_[0]),
      perm_host_.size(),
      false);
  cudaDeviceSynchronize();

  // compute prox with permuted argument
  base_prox_->EvalLocal(
    permuted_arg_.begin(), 
    permuted_arg_.end(),
    result_beg,
    result_end,
    tau_beg,
    tau_end,
    tau, 
    invert_tau);

  // permute result back
  ProxPermuteKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&permuted_arg_[0]),
      thrust::raw_pointer_cast(&perm_[0]),
      perm_host_.size(),
      true);
  cudaDeviceSynchronize();
}

template<typename T>
size_t ProxPermute<T>::gpu_mem_amount() const 
{
  return this->size_ * sizeof(T) + base_prox_->gpu_mem_amount();
}

template<typename T>
void ProxPermute<T>::get_separable_structure(
  vector<std::tuple<size_t, size_t, size_t> >& sep)
{
  base_prox_->get_separable_structure(sep);
}


// Explicit template instantiation
template class ProxPermute<float>;
template class ProxPermute<double>;

} // namespace prost
