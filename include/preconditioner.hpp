#ifndef PRECONDITIONER_HPP_
#define PRECONDITIONER_HPP_

#include <vector>

#include "config.hpp"
#include "prox.hpp"
#include "util/sparse_matrix.hpp"

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
  Preconditioner(SparseMatrix<real> *mat);
  virtual ~Preconditioner();

  void ComputeScalar();
  void ComputeAlpha(real alpha);
  void ComputeEquil();

  // renormalizes S and T in such a way that ||S^(1/2) K T^(1/2)|| <= 1
  // and that prox-operators which can only handle scalar steps have the
  // same step size along the according entries in the diagonal matrix. 
  void Renormalize(
      const std::vector<Prox *>& prox_g,
      const std::vector<Prox *>& prox_hc);

  real *left() const { return d_left_; }
  real *right() const { return d_right_; }
  PreconditionerType precond_type() const { return type_; }
  int gpu_mem_amount() {
    return (mat_->nrows() + mat_->ncols()) * sizeof(real);
  }

private:
  SparseMatrix<real> *mat_;
  real *d_left_;
  real *d_right_;
  PreconditionerType type_;
};

#endif
