#ifndef ELEM_OPERATION_NORM2_HPP_
#define ELEM_OPERATION_NORM2_HPP_

#include "elem_operation.hpp"
#include "function_1d.hpp"
/**
 * @brief Provides proximal operator for sum of 2-norms, with a nonlinear
 *        function ProxFunction1D applied to the norm.
 *
 *
 *
 */
namespace prox {
namespace elemOperation {
template<typename T, size_t DIM, FUN_1D>
struct ElemOperationNorm2 : public ElemOperation<DIM> {
 struct Coefficients : public Coefficients1D<T> {};

 ElemOperationNorm2(Coefficients& coeffs) : coeffs_(coeffs) {} 
 
 
 inline __device__ void operator()(Vector<T, ElemOperationNorm2>& arg, Vector<T, ElemOperationNorm2>& res, Vector<T,ElemOperationNorm2>& tau_diag, T tau_scal, bool invert_tau, SharedMem<ElemOperationNorm2>& shared_mem) {
    // compute dim-dimensional 2-norm at each point
    T norm = 0;
    size_t index;

    for(size_t i = 0; i < dim; i++) {
      const T arg = arg[i];
      norm += arg * arg;
    }
    
    //TODO check coeffs_.val[i] == NULL
    
    if(norm > 0) {
      norm = sqrt(norm);

      // compute step-size
      tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);

      // compute scaled prox argument and step 
      const T arg = ((coeffs_.a * (norm - coeffs_.d * tau)) /
                     (1. + tau * coeffs_.e)) - coeffs_.b;
    
      const T step = (coeffs_.c * coeffs_.a * coeffs_.a * tau) /
                     (1. + tau * coeffs_.e);
      
      // compute prox
      FUN_1D fun;
      const T prox_result = (fun(arg, step, coeffs_.alpha, coeffs_.beta) +
                             coeffs_.b) / coeffs_.a;

      // combine together for result
      for(size_t i = 0; i < dim; i++) {
        res[i] = prox_result * arg[i] / norm;
      }
    } else { // in that case, the result is zero. 
      for(size_t i = 0; i < dim; i++) {
        res[i] = 0;
      }
    }
 }
}
}
#endif
