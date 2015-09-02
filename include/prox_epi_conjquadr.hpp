#ifndef PROX_EPI_CONJQUADR_HPP_
#define PROX_EPI_CONJQUADR_HPP_

#include <vector>

#include "prox.hpp"

struct EpiConjQuadrCoeffs {
  static int num_coeffs() {
    return 5;
  }
  
  std::vector<real> a, b, c, alpha, beta;
};

/**
 * @brief Computes orthogonal projection of (u,v) onto the convex set
 *        C = { (x, y) | y >= (ax^2 + bx + c + delta(alpha <= x <= beta))* },
 *        where * denotes the Legendre-Fenchel conjugate.
 */
class ProxEpiConjQuadr : public Prox {
public:
  ProxEpiConjQuadr(
      int index,
      int count,
      bool interleaved,
      const EpiConjQuadrCoeffs& coeffs);

  virtual ~ProxEpiConjQuadr();

  virtual void Evaluate(
      real *d_arg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_step = false);

protected:
  EpiConjQuadrCoeffs coeffs_;
  std::vector<real *> d_coeffs_;
};

#endif
