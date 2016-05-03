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

#ifndef PROST_ELEM_OPERATION_1D_HPP_
#define PROST_ELEM_OPERATION_1D_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

// TODO: comment me
template<typename T, class FUN_1D>
struct ElemOperation1D : public ElemOperation<1, 7> 
{
    
  inline __host__ __device__
  ElemOperation1D(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) : coeffs_(coeffs) { } 
  
  inline __host__ __device__
  void operator()(Vector<T>& res, const Vector<const T>& arg, const Vector<const T>& tau_diag, T tau_scal, bool invert_tau)
  {
    // compute step-size
    T tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);

    if(coeffs_[0] == 0 || coeffs_[2] == 0) {
      res[0] = (arg[0] - tau * coeffs_[3]) / (1 + tau * coeffs_[4]);
    }
    else {
      // compute scaled prox argument and step 
      const T prox_arg = ((coeffs_[0] * (arg[0] - coeffs_[3] * tau)) /
        (1. + tau * coeffs_[4])) - coeffs_[1];

      const T step = (coeffs_[2] * coeffs_[0] * coeffs_[0] * tau) /
        (1. + tau * coeffs_[4]);

      // compute scaled prox and store result
      FUN_1D fun;
      res[0] =
        (fun(prox_arg, step, coeffs_[5], coeffs_[6]) + coeffs_[1])
        / coeffs_[0];
    }
  }
  
private:
  T* coeffs_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_1D_HPP_

