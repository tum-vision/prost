/*
 * This file is part of pdsolver.
 *
 * Copyright (C) 2015 Thomas Möllenhoff <thomas.moellenhoff@in.tum.de> 
 *
 * pdsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pdsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pdsolver. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PDSOLVER_BACKEND_H
#define PDSOLVER_BACKEND_H

#include <string>
#include <vector>

#include "../prox/prox.h"
#include "../../config.h"
#include "../../utils/sparse_matrix.h"

enum backend_type
{
  kBackendPDHG,
};

enum pdhg_type
{
  kAlg1,
  kAlg2,
  kAdapt,
};

enum precond_type
{
  kPrecondOff,
  kPrecondDiagonal,
  kPrecondEquilibrate
};

struct solver_options {
  backend_type type;

  int max_iters;
  int cb_iters;
  real tolerance;

  // parameters for preconditioner
  precond_type precond;
  real precond_alpha;

  // parameters for primal-dual
  pdhg_type pdhg;
  real gamma;
  real alpha0, nu, delta, s;  
};

class backend {
public:
  backend(sparse_matrix<real> *K,
          real *d_lft, real *d_rgt,
          const std::vector<prox *>& pg,
          const std::vector<prox *>& phc,
          const solver_options& o)
      : mat(K), d_precond_lft(d_lft), d_precond_rgt(d_rgt),
      prox_g(pg), prox_hc(phc), opts(o) {}
  
  virtual ~backend() {}

  virtual bool initialize() = 0;
  virtual void do_iteration() = 0;
  virtual void get_iterates(real *x, real *y) = 0;
  virtual bool is_converged() = 0;
  virtual void release() = 0;

  virtual std::string status() = 0;

protected:
  int nrows, ncols;
  sparse_matrix<real> *mat;
  std::vector<prox *> prox_g;
  std::vector<prox *> prox_hc;
  real *d_precond_lft;
  real *d_precond_rgt;
  solver_options opts;
};

#endif
