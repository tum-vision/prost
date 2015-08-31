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

#ifndef PDSOLVER_H
#define PDSOLVER_H

#include <vector>

#include "prox/prox.h"
#include "backend/backend.h"
#include "../utils/sparse_matrix.h"
#include "../config.h"

typedef void (*pdsolver_callback_ptr)(int, real*, real*, bool);

class pdsolver {
public:
  pdsolver();
  virtual ~pdsolver();
  
  // set data
  void set_matrix_csc(real *val, int *ptr, int *ind, int m, int n, int nnz);

  // important: add_prox_... passes ownership to pdsolver
  void add_prox_g(prox *p);
  void add_prox_hc(prox *p);
  
  void set_backend(const solver_options& opts);
  void set_callback(pdsolver_callback_ptr cb); 

  // algorithm
  bool initialize();
  void solve();
  void release();

  real *get_primal_iterate() const { return h_x; }
  real *get_dual_iterate() const { return h_y; }
  
protected:
  std::vector<prox *> prox_g;
  std::vector<prox *> prox_hc;
  sparse_matrix<real> *mat;
  real *d_precond_lft;
  real *d_precond_rgt;
  backend *be;
  solver_options opts;
  real *h_x, *h_y; // primal and dual iterates on host
  pdsolver_callback_ptr callback;
};

#endif






