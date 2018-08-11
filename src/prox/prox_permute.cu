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
#include "prost/exception.hpp"

namespace prost {

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
  // TODO: permute argument

  // compute prox with scaled argument
  base_prox_->EvalLocal(
    result_beg, 
    result_end,
    permuted_arg_.begin(),
    permuted_arg_.end(),
    tau_beg,
    tau_end,
    tau, 
    !invert_tau);

  // TODO: permute result
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
