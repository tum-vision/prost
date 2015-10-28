#ifndef PROX_EPI_CONJQUADR_SCALED_HPP_
#define PROX_EPI_CONJQUADR_SCALED_HPP_

#include <vector>

#include "prox.hpp"
#include "prox_epi_conjquadr.hpp"

/**
 * @brief Computes orthogonal projection of (u, v) onto the convex set
 *        C = { (x, y) | y >= (ax^2 + bx + c + delta(alpha <= x <= beta))* },
 *        where * denotes the Legendre-Fenchel conjugate.
 *
 *        Prox is scaled by factor scaling.
 */
template<typename T>
class ProxEpiConjQuadrScaled : public Prox<T> {
 public:
  ProxEpiConjQuadrScaled(size_t index,
    size_t count,
    bool interleaved,
    const EpiConjQuadrCoeffs<T>& coeffs,
    T scaling);

  virtual ~ProxEpiConjQuadrScaled();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  T scaling_;
  EpiConjQuadrCoeffs<T> coeffs_;
  EpiConjQuadrCoeffsDevice<T> coeffs_dev_;
};

#endif
