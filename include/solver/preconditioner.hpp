#ifndef PRECONDITIONER_HPP_
#define PRECONDITIONER_HPP_

#include <vector>

#include "config.hpp"
#include "prox/prox.hpp"
#include "linop/linop.hpp"

enum PreconditionerType {
  kPrecondScalar, // S = T = (1 / |K|)
  kPrecondAlpha, // Pock, Chambolle, ICCV '11
  kPrecondEquil, // Matrix Equilibration
};

/**
 * @brief Diagonal preconditioners.
 *
 */
class Preconditioner {
public:
  Preconditioner(LinearOperator<real> *linop);
  virtual ~Preconditioner();

  void ComputeScalar();
  void ComputeAlpha(
      real alpha,
      const std::vector<Prox<real> *>& prox_g,
      const std::vector<Prox<real> *>& prox_hc);
  void ComputeEquil();

  real *left() const { return d_left_; }
  real *right() const { return d_right_; }
  PreconditionerType precond_type() const { return type_; }
  int gpu_mem_amount() {
    return (linop_->nrows() + linop_->ncols()) * sizeof(real);
  }

private:
  LinearOperator<real> *linop_;
  real *d_left_;
  real *d_right_;
  PreconditionerType type_;
};

#endif
