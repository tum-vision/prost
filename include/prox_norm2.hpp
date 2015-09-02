#ifndef PROX_NORM2_HPP_
#define PROX_NORM2_HPP_

#include "prox_1d.hpp"

/**
 * @brief Provides proximal operator for sum of 2-norms, with a nonlinear
 * function ProxFunction1D applied to the norm.
 */
class ProxNorm2 : public Prox1D {
public:
  ProxNorm2(
      int index,
      int count,
      int dim,
      bool interleaved,
      const Prox1DCoefficients& coeffs,
      const Prox1DFunction& func);
  
  virtual ~ProxNorm2();

  virtual void Evaluate(
      real *d_proxarg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_step = false);
};

#endif
