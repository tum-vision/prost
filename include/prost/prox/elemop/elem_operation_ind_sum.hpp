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

#ifndef PROST_ELEM_OPERATION_IND_SUM_HPP_
#define PROST_ELEM_OPERATION_IND_SUM_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

/// 
/// \brief Computes prox for sum of sum-to-one indicator functions.
///
template<typename T>
struct ElemOperationIndSum : public ElemOperation<0, 0, T>
{
  __device__
  ElemOperationIndSum(size_t dim, SharedMem<typename ElemOperationIndSum::SharedMemType, typename ElemOperationIndSum::GetSharedMemCount>& shared_mem)
      : dim_(dim) { } 
  
  inline __device__
  void
  operator()(
    Vector<T>& res,
    const Vector<const T>& arg,
    const Vector<const T>& tau_diag,
    T tau_scal,
    bool invert_tau) 
  {
    T tl = 0;

    for(size_t i = 0; i < dim_; i++) {
      tl += arg[i];
    }

    tl = (tl - 1.) / static_cast<T>(dim_);

    for(size_t i = 0; i < dim_; i++) {
      res[i] = arg[i] - tl; 
    }
  }
    
 private:
  size_t dim_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_IND_SUM_HPP_
