#ifndef PROX_SIMPLEX_HPP_
#define PROX_SIMPLEX_HPP_

#include <vector>
#include "prox.hpp"

/**
 * @brief Computes prox for sum of simplex indicator functions
 *        plus a linear term:
 *
 *        sum_{i=1}^{count} delta_dim(x_i) + <x_i, a_i>,
 *
 *        where delta_dim denotes the dim-dimensional simplex.
 *        See http://arxiv.org/pdf/1101.6081v2.pdf.
 */
class ProxSimplex : public Prox {
public:
  ProxSimplex(
      int index,
      int count,
      int dim,
      bool interleaved,
      const std::vector<real>& coeffs);

  virtual ~ProxSimplex();

  virtual void Evaluate(
      real *d_arg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_step = false);

  virtual int gpu_mem_amount();

protected:
  std::vector<real> coeffs_;
  real *d_coeffs_;
};

#endif
