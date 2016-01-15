#ifndef ELEM_OPERATION_1D_HPP_
#define ELEM_OPERATION_1D_HPP_

#include <string>
#include <vector>

/*
enum Prox1DFunction {
  kZero = 0,           // 0
  kAbs,                // |x|         
  kSquare,             // (1/2) x^2
  kMaxPos0,            // max(0, x)
  kIndLeq0,            // delta(x<=0) 
  kIndGeq0,            // delta(x>=0)
  kIndEq0,             // delta(x=0)
  kIndBox01,           // delta(0<=x<=1)
  kL0,                 // |x|^0
  kHuber,              // huber(x)
  
  kNumProx1DFunctions,
  kInvalidProx = -1
};

#define PROX_1D_NUM_COEFFS 7

template<typename T>
struct Prox1DCoefficients {
  std::vector<T> a, b, c, d, e;
  std::vector<T> alpha, beta;
};

template<typename T>
struct Prox1DCoeffsDevice {
  T *d_ptr[PROX_1D_NUM_COEFFS];
  T val[PROX_1D_NUM_COEFFS];
};
*/

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
template<typename T, class OPERATION_1D>
struct ElemOperation1D {
 struct Data {
    T a, b, c, d, e, alpha, beta;
 };
 
 static const size_t dim = 1;

 inline __device__ void operator()(ProxSeparableSum<T, ElemOperation1D>::Vector& arg, ProxSeparableSum<T, ElemOperation1D>::Vector& res, Data& data) { res[0] = do something with OPERATION_1D.Eval(arg[0], data.alpha, data.beta); }
};
}
}

#endif
