#ifndef PROX_1D_HPP_
#define PROX_1D_HPP_

#include "prox.hpp"

#include <string>
#include <vector>

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

/**
 * @brief Provides proximal operator for fully separable 1D functions:
 * 
 *        sum_i c_i * f(a_i x - b_i) + d_i x + (e_i / 2) x^2.
 *
 *        alpha and beta are generic parameters depending on the choice of f,
 *        e.g. the power for f(x) = |x|^{alpha}.
 *
 */
template<typename T>
class Prox1D : public Prox<T> {
public:
  Prox1D(size_t index,
         size_t count,
         const Prox1DCoefficients<T>& coeffs,
         const Prox1DFunction& func);
  
  virtual ~Prox1D();

  virtual bool Init();
  virtual void Release();
  
  virtual size_t gpu_mem_amount();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);

  Prox1DFunction func_;
  Prox1DCoefficients<T> coeffs_;
  Prox1DCoeffsDevice<T> coeffs_dev_;
};

// implementation of 1D prox operators
#ifdef __CUDACC__ 

template<typename T>
struct Prox1DZero {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    return x0;
  }
};

template<typename T> 
struct Prox1DAbs {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    if(x0 >= tau)
      return x0 - tau;
    else if(x0 <= -tau)
      return x0 + tau;

    return 0;
  }
};

template<typename T>
struct Prox1DSquare {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    return x0 / (1. + tau);
  }
};

template<typename T>
struct Prox1DIndLeq0 {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    if(x0 > 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Prox1DIndGeq0 {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Prox1DIndEq0 {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    return 0.;
  }
};

template<typename T>
struct Prox1DIndBox01 {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    if(x0 > 1.)
      return 1.;
    else if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Prox1DMaxPos0 {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    if(x0 > tau)
      return x0 - tau;
    else if(x0 < 0.)
      return x0;

    return 0.;
  }
};

template<typename T>
struct Prox1DL0 {
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    if(x0*x0 > 2 * tau)
      return x0;

    return 0;
  }
};

template<typename T>
struct ProxHuber {
  // min_x huber_alpha(x) + (1/2tau) (x-x0)^2
  inline __device__ T Eval(T x0, T tau, T alpha, T beta) const {
    T result = (x0 / tau) / (static_cast<T>(1) + alpha / tau);
    result /= max(static_cast<T>(1), abs(result));  
    return x0 - tau * result;
  }
};

#endif

#endif
