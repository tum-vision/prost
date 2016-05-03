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

#ifndef PROST_ELEM_OPERATION_NORM2_HPP_
#define PROST_ELEM_OPERATION_NORM2_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

/// 
/// \brief Provides proximal operator for sum of 2-norms, with a nonlinear
///        function Function1D applied to the norm.
/// 
template<typename T, class FUN_1D>
struct ElemOperationNorm2 : public ElemOperation<0, 7> 
{
  __host__ __device__ 
  ElemOperationNorm2(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) 
    : coeffs_(coeffs), dim_(dim) { } 
 
 inline __host__ __device__ 
 void operator()(
     Vector<T>& res, 
     const Vector<const T>& arg, 
     const Vector<const T>& tau_diag, 
     T tau_scal, 
     bool invert_tau) 
  {
    // compute dim-dimensional 2-norm at each point
    T norm = 0;

    for(size_t i = 0; i < dim_; i++)
    {
      const T val = arg[i];
      norm += val * val;
    }

    if(norm > 0)
    {
      norm = sqrt(norm);

      // compute step-size
      T tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);

      // compute scaled prox argument and step 
      const T prox_arg = ((coeffs_[0] * (norm - coeffs_[3] * tau)) /
                     (1. + tau * coeffs_[4])) - coeffs_[1];

      const T step = (coeffs_[2] * coeffs_[0] * coeffs_[0] * tau) /
                     (1. + tau * coeffs_[4]);

      // compute prox
      FUN_1D fun;
      const T prox_result = (fun(prox_arg, step, coeffs_[5], coeffs_[6]) +
                             coeffs_[1]) / coeffs_[0];

      // combine together for result
      for(size_t i = 0; i < dim_; i++)
      {
        res[i] = prox_result * arg[i] / norm;
      }
    }
    else
    { // in that case, the result is zero. 
      for(size_t i = 0; i < dim_; i++)
      {
        res[i] = 0;
      }
    }    
 }
 
private:
  T* coeffs_;
  size_t dim_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_NORM2_HPP_
