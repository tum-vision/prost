#ifndef PROX_NORM2_HPP_
#define PROX_NORM2_HPP_

#include "prox_1d.hpp"

/**
 * @brief Provides proximal operator for sum of 2-norms, with a nonlinear
 *        function ProxFunction1D applied to the norm.
 *
 */
template<typename T>
class ProxNorm2 : public Prox1D<T> {
 public:
  ProxNorm2(size_t index,
            size_t count,
            size_t dim,
            bool interleaved,
            const Prox1DCoefficients& coeffs,
            const Prox1DFunction& func);
  
  virtual ~ProxNorm2();

 protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         real tau,
                         bool invert_tau);
};

#endif
