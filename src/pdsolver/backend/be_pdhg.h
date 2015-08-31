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

#ifndef PDSOLVER_BE_PDHG
#define PDSOLVER_BE_PDHG

#include <cublas_v2.h>
#include "backend.h"

class be_pdhg : public backend {
public:
  be_pdhg(sparse_matrix<real> *K,
          real *d_lft, real *d_rgt,
          const std::vector<prox *>& pg,
          const std::vector<prox *>& phc,
          const solver_options& o)
      : backend(K, d_lft, d_rgt, pg, phc, o) { }
  virtual ~be_pdhg() { }
  
  virtual bool initialize();
  virtual void do_iteration();
  virtual void get_iterates(real *x, real *y);
  virtual bool is_converged();
  virtual void release();

  virtual std::string status();
  
protected:
  int m, n, l;

  cublasHandle_t cublas_handle;
  
  // algorithm variables
  real *d_x, *d_y;
  real *d_prox_arg;
  real *d_x_prev, *d_y_prev;
  real *d_x_mat, *d_y_mat;
  real *d_x_mat_prev, *d_y_mat_prev;
  real *d_res_primal, *d_res_dual;
  real tau, sigma, theta;
  real res_primal, res_dual;
};

#endif
