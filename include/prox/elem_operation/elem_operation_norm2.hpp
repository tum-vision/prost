#ifndef ELEM_OPERATION_NORM2_HPP_
#define ELEM_OPERATION_NORM2_HPP_

#include "prox_1d.hpp"

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
 struct Coefficients {
    T a, b, c, d, e, alpha, beta;
 };

 typedef Vec ProxSeparableSum<T, ElemOperationNorm2>::Vector;
 
 inline __device__ void operator()(Vec& d_arg, Vec& d_res, Vec& d_tau, T tau, Coefficients& coeffs, bool invert_tau) {
    // compute dim-dimensional 2-norm at each point
    T norm = 0;
    size_t index;

    for(size_t i = 0; i < dim; i++) {
      index = interleaved ? (tx * dim + i) : (tx + count * i);
      const T arg = arg[index];
      norm += arg * arg;
    }
    
    if(norm > 0) {
      norm = sqrt(norm);

      // compute step-size
      tau = invert_tau ? (1. / (tau * d_tau[0])) : (tau * d_tau[0]);

      // compute scaled prox argument and step 
      const T arg = ((coeffs.a * (norm - coeffs.d * tau)) /
                     (1. + tau * coeffs.e)) - coeffs.b;
    
      const T step = (coeffs.c * coeffs.a * coeffs.a * tau) /
                     (1. + tau * coeffs.e);
      
      // compute prox
      const T prox_result = (FUN_1D(arg, step, coeffs.alpha, coeffs.beta) +
                             coeffs.b) / coeffs.a;

      // combine together for result
      for(size_t i = 0; i < dim; i++) {
        d_res[i] = prox_result * d_arg[i] / norm;
      }
    } else { // in that case, the result is zero. 
      for(size_t i = 0; i < dim; i++) {
        d_res[i] = 0;
      }
    }
 }
}
}
#endif
