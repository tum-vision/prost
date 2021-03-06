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

#ifndef PROST_PROX_IND_SUM_HPP_
#define PROST_PROX_IND_SUM_HPP_

#include <thrust/device_vector.h>

#include "prost/prox/prox.hpp"
#include "prost/common.hpp"

namespace prost {

/// 
/// \brief Evaluates the prox of function composed with a permutation matrix
/// 
template<typename T>
class ProxIndSum : public Prox<T> {
public:
  ProxIndSum(
      size_t index, 
      size_t size,
      size_t count,
      size_t dim,
      std::vector<size_t>& inds,
      T sum)
    : Prox<T>(index, size, true) { inds_ = inds; dim_ = dim; count_ = count; two_ = false; sum_ = sum; }

  ProxIndSum(size_t index, 
	     size_t size,
	     size_t count,
	     size_t dim, 
	     std::vector<size_t>& inds,
	     T sum,
	     size_t count2,
	     size_t dim2,
	     std::vector<size_t>& inds2,
	     T sum2)
    : Prox<T>(index, size, true) { inds_ = inds; dim_ = dim; count_ = count; two_ = true; dim_2_ = dim2; count_2_ = count2; inds_2_ = inds2; sum_ = sum; sum_2_ = sum2; }

  
  virtual ~ProxIndSum() {}

  virtual void Initialize();
  virtual void Release();

  virtual size_t gpu_mem_amount() const;

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
  size_t dim_, dim_2_;
  size_t count_, count_2_;
  std::vector<size_t> inds_, inds_2_;
  device_vector<size_t> d_inds_, d_inds_2_;
  bool two_;
  T sum_, sum_2_;
};

} // namespace prost

#endif // PROST_PROX_IND_SUM_HPP_
