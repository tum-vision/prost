#ifndef FUNCTION_1D_HPP_
#define FUNCTION_1D_HPP_

#include <string>
#include <vector>

// implementation of 1D prox operators
#ifdef __CUDACC__ 

template<typename T>
struct Coefficients1D {
    T a, b, c, d, e, alpha, beta;
};
    
template<typename T>
struct Function1DZero {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    return x0;
  }
};

template<typename T> 
struct Function1DAbs {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 >= tau)
      return x0 - tau;
    else if(x0 <= -tau)
      return x0 + tau;

    return 0;
  }
};

template<typename T>
struct Function1DSquare {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    return x0 / (1. + tau);
  }
};

template<typename T>
struct Function1DIndLeq0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 > 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Function1DIndGeq0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Function1DIndEq0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    return 0.;
  }
};

template<typename T>
struct Function1DIndBox01 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 > 1.)
      return 1.;
    else if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Function1DMaxPos0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 > tau)
      return x0 - tau;
    else if(x0 < 0.)
      return x0;

    return 0.;
  }
};

template<typename T>
struct Function1DL0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0*x0 > 2 * tau)
      return x0;

    return 0;
  }
};

template<typename T>
struct Function1DHuber {
  // min_x huber_alpha(x) + (1/2tau) (x-x0)^2
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    T result = (x0 / tau) / (static_cast<T>(1) + alpha / tau);
    result /= max(static_cast<T>(1), abs(result));  
    return x0 - tau * result;
  }
};

#endif

#endif
