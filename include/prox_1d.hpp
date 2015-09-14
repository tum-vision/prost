#ifndef PROX_1D_HPP_
#define PROX_1D_HPP_

#include "prox.hpp"

#include <string>
#include <vector>

enum Prox1DFunction {
  kZero = 0,               // 0      
  kAbs,                // |x|         
  kSquare,             // (1/2) x^2
  kIndicatorLeq,       // delta(x<=1) TODO: replace this to x<=0?
  kIndicatorGeq,       // delta(x>=0)
  kIndicatorEq,        // delta(x=0)
  kIndicatorAbsLeq,    // delta(-1<=x<=1)
  
  kNumProx1DFunctions,
  kInvalidProx = -1
};

struct Prox1DCoefficients {
  static int num_coeffs() {
    return 5;
  }
  
  std::vector<real> a, b, c, d, e;
};

/**
 * @brief Provides proximal operator for fully separable functions:
 * 
 *   sum_i c_i * f(a_i x - b_i) + d_i x + e_i x^2.
 *
 */
class Prox1D : public Prox {
public:
  Prox1D(
      int index,
      int count,
      const Prox1DCoefficients& coeffs,
      const Prox1DFunction& func);

  virtual ~Prox1D();

  virtual void Evaluate(
      real *d_arg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_step = false);

  virtual int gpu_mem_amount();
  
protected:
  Prox1DFunction func_;
  Prox1DCoefficients coeffs_;
  std::vector<real *> d_coeffs_;
};

extern "C" Prox1DFunction Prox1DFunctionFromString(std::string name);

// implementation of 1D prox operators
#ifdef __CUDACC__ 

template<typename real>
struct Prox1DZero {
  inline __device__ real Apply(real x0, real tau) const {
    return x0;
  }
};

template<typename real> 
struct Prox1DAbs {
  inline __device__ real Apply(real x0, real tau) const {
    if(x0 >= tau)
      return x0 - tau;
    else if(x0 <= -tau)
      return x0 + tau;

    return 0;
  }
};

template<typename real>
struct Prox1DSquare {
  inline __device__ real Apply(real x0, real tau) const {
    return x0 / (1. + tau);
  }
};

template<typename real>
struct Prox1DIndicatorLeq
{
  inline __device__ real Apply(real x0, real tau) const {
    if(x0 > 1.)
      return 1.;

    return x0;
  }
};

template<typename real>
struct Prox1DIndicatorGeq
{
  inline __device__ real Apply(real x0, real tau) const {
    if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename real>
struct Prox1DIndicatorEq {
  inline __device__ real Apply(real x0, real tau) const {
    return 0.;
  }
};

template<typename real>
struct Prox1DIndicatorAbsLeq
{
  inline __device__ real Apply(real x0, real tau) const {
    if(x0 > 1.)
      return 1.;
    else if(x0 < -1.)
      return -1.;

    return x0;
  }
};

#endif

#endif
