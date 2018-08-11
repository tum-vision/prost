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

#ifndef PROST_PROX_PERMUTE_HPP_
#define PROST_PROX_PERMUTE_HPP_

#include <thrust/device_vector.h>

#include "prost/prox/prox.hpp"
#include "prost/common.hpp"

namespace prost {

/// 
/// \brief Evaluates the prox of function composed with a permutation matrix
/// 
template<typename T>
class ProxPermute : public Prox<T> {
public:
  ProxPermute(shared_ptr<Prox<T>> base_prox, const std::vector<int>& perm);
  virtual ~ProxPermute();

  virtual void Initialize();
  virtual void Release();

  virtual size_t gpu_mem_amount() const;
  virtual void get_separable_structure(vector<std::tuple<size_t, size_t, size_t> >& sep);

protected:
  virtual void EvalLocal(
    const typename device_vector<T>::iterator& result_beg,
    const typename device_vector<T>::iterator& result_end,
    const typename device_vector<T>::const_iterator& arg_beg,
    const typename device_vector<T>::const_iterator& arg_end,
    const typename device_vector<T>::const_iterator& tau_beg,
    const typename device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);

private:
  shared_ptr<Prox<T>> base_prox_;
  device_vector<T> permuted_arg_;
  std::vector<int> perm_host_;
  device_vector<int> perm_;
};

} // namespace prost

#endif // PROST_PROX_PERMUTE_HPP_
