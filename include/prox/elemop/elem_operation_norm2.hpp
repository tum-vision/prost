#ifndef ELEM_OPERATION_NORM2_HPP_
#define ELEM_OPERATION_NORM2_HPP_

#include "elem_operation.hpp"

/// 
/// \brief Provides proximal operator for sum of 2-norms, with a nonlinear
///        function Function1D applied to the norm.
/// 
template<typename T, class FUN_1D>
struct ElemOperationNorm2 : public ElemOperation<0, 7> 
{
  __host__ __device__ 
  ElemOperationNorm2(T* coeffs, size_t dim, SharedMem<ElemOperationNorm2<T, FUN_1D>>& shared_mem) 
    : coeffs_(coeffs), dim_(dim) { } 
 
 inline __host__ __device__ 
 void operator()(
     Vector<T>& res, 
     const Vector<T>& arg, 
     const Vector<T>& tau_diag, 
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

#endif
