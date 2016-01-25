#ifndef ELEM_OPERATION_1D_HPP_
#define ELEM_OPERATION_1D_HPP_

#include "elem_operation.hpp"
#include "function_1d.hpp"
/**
 * @brief Provides proximal operator for fully separable 1D functions:
 * 
 *        sum_i c_i * f(a_i x - b_i) + d_i x + (e_i / 2) x^2.
 *
 *        alpha and beta are generic parameters depending on the choice of f,
 *        e.g. the power for f(x) = |x|^{alpha}.
 *
 */

template<typename T, class FUN_1D>
struct ElemOperation1D : public ElemOperation<1> {
  struct Coefficients : public Coefficients1D<T> {};
   __device__ ElemOperation1D() {} 
 
  __device__ ElemOperation1D(Coefficients& coeffs) : coeffs_(coeffs) {} 
  
  inline __device__ void operator()(Vector<T, ElemOperation1D>& arg, Vector<T, ElemOperation1D>& res, Vector<T, ElemOperation1D>& tau_diag, T tau_scal, bool invert_tau, SharedMem<ElemOperation1D>& shared_mem) {

    //TODO check coeffs_.val[i] == NULL

    if(coeffs_->c == 0) // c == 0 -> prox_zero -> return argument
      res[0] = arg[0];
    else {
      // compute step-size
      const T tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);

      // compute scaled prox argument and step 
      const T prox_arg = ((coeffs_->a * (arg[0] - coeffs_->d * tau)) /
        (1. + tau * coeffs_->e)) - coeffs_->b;
    
      const T step = (coeffs_->c * coeffs_->a * coeffs_->a * tau) /
        (1. + tau * coeffs_->e);

      // compute scaled prox and store result
      FUN_1D fun;
      res[0] = 
        (fun(prox_arg, step, coeffs_->alpha, coeffs_->beta) + coeffs_->b)
        / coeffs_->a;
    }
  }
  
private:
  Coefficients* coeffs_;
};


#endif
