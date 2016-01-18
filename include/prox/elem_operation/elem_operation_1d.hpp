#ifndef ELEM_OPERATION_1D_HPP_
#define ELEM_OPERATION_1D_HPP_


/**
 * @brief Provides proximal operator for fully separable 1D functions:
 * 
 *        sum_i c_i * f(a_i x - b_i) + d_i x + (e_i / 2) x^2.
 *
 *        alpha and beta are generic parameters depending on the choice of f,
 *        e.g. the power for f(x) = |x|^{alpha}.
 *
 */
namespace prox {
namespace elemOperation {
template<typename T, class FUN_1D>
struct ElemOperation1D : public ElemOperation<1> {
  struct Coefficients {
    T a, b, c, d, e, alpha, beta;
  };

  typedef Vec ProxSeparableSum<T, ElemOperation1D>::Vector;
 
 
  inline __device__ void operator()(Vec& d_arg, Vec& d_res, Vec& d_tau, T tau, Coefficients& coeffs, bool invert_tau) {

    //TODO check coeffs.val[i] == NULL

    if(coeffs.c == 0) // c == 0 -> prox_zero -> return argument
      d_res[0] = d_arg[0];
    else {
      // compute step-size
      tau = invert_tau ? (1. / (tau * d_tau[0])) : (tau * d_tau[0]);

      // compute scaled prox argument and step 
      const T arg = ((coeffs.a * (d_arg[0] - coeffs.d * tau)) /
        (1. + tau * coeffs.e)) - coeffs.b;
    
      const T step = (coeffs.c * coeffs.a * coeffs.a * tau) /
        (1. + tau * coeffs.e);

      // compute scaled prox and store result
      d_res[0] = 
        (FUN_1D(arg, step, coeffs.alpha, coeffs.beta) + coeffs.b)
        / coeffs.a;
    }
 }
};
}
}

#endif
