#ifndef OPERATION_1D_HPP_
#define OPERATION_1D_HPP_

#include <string>
#include <vector>

// implementation of 1D prox operators
#ifdef __CUDACC__ 

template<typename T>
struct Operation1DZero {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    return x0;
  }
};

template<typename T> 
struct Operation1DAbs {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 >= tau)
      return x0 - tau;
    else if(x0 <= -tau)
      return x0 + tau;

    return 0;
  }
};

template<typename T>
struct Operation1DSquare {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    return x0 / (1. + tau);
  }
};

template<typename T>
struct Operation1DIndLeq0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 > 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Operation1DIndGeq0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Operation1DIndEq0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    return 0.;
  }
};

template<typename T>
struct Operation1DIndBox01 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 > 1.)
      return 1.;
    else if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Operation1DMaxPos0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0 > tau)
      return x0 - tau;
    else if(x0 < 0.)
      return x0;

    return 0.;
  }
};

template<typename T>
struct Operation1DL0 {
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    if(x0*x0 > 2 * tau)
      return x0;

    return 0;
  }
};

template<typename T>
struct Operation1DHuber {
  // min_x huber_alpha(x) + (1/2tau) (x-x0)^2
  inline __device__ T operator()(T x0, T tau, T alpha, T beta) const {
    T result = (x0 / tau) / (static_cast<T>(1) + alpha / tau);
    result /= max(static_cast<T>(1), abs(result));  
    return x0 - tau * result;
  }
};

#endif

#endif
