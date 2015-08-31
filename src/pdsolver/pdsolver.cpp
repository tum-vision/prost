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

#include "pdsolver.h"

#include <iostream>
#include "backend/be_pdhg.h"

pdsolver::pdsolver() {
  
}

pdsolver::~pdsolver() {
}
  
void pdsolver::set_matrix_csc(real *val, int *ptr, int *ind, int m, int n, int nnz) {
  mat = sparse_matrix<real>::create_from_csc(m, n, nnz, val, ptr, ind);
}

void pdsolver::add_prox_g(prox *p) {
  prox_g.push_back(p);
}

void pdsolver::add_prox_hc(prox *p) {
  prox_hc.push_back(p);
}

void pdsolver::set_backend(const solver_options& opts) {
  this->opts = opts;
}

void pdsolver::set_callback(pdsolver_callback_ptr cb) {
  callback = cb;
}

bool pdsolver::initialize() {
  // @TODO build preconditioners
  //   T = column sum of |K|^(2-alpha)
  //   S = row sum of |K|^alpha
  //   loop over prox operators and average coupled entries in T and S
  //   form matrix M = sqrt(S) * K * sqrt(T)
  //   norm=normest(M)
  //   T = T / norm
  //   S = S / norm
  
  // for now: set to scaled identity so that norm(S^0.5 * K * T^0.5)^2 = 1
  real norm = normest(*mat);
  std::cout << "Operator norm = " << norm << std::endl;

  int m = mat->nrows();
  int n = mat->ncols();
  
  // allocate primal/dual iterates on host
  h_x = new real[n];
  h_y = new real[m];

  // build most simple diagonal preconditioners (constant everywhere, c*I)
  real *h_pc_lft = new real[m];
  real *h_pc_rgt = new real[n];
  for(int i = 0; i < m; i++) h_pc_lft[i] = 1.0 / norm;
  for(int i = 0; i < n; i++) h_pc_rgt[i] = 1.0 / norm;

  // copy preconditioners to GPU
  cudaMalloc((void **)&d_precond_lft, sizeof(real) * m);
  cudaMalloc((void **)&d_precond_rgt, sizeof(real) * n);
  cudaMemcpy(d_precond_lft, h_pc_lft, sizeof(real) * m, cudaMemcpyHostToDevice);
  cudaMemcpy(d_precond_rgt, h_pc_rgt, sizeof(real) * n, cudaMemcpyHostToDevice);

  // cleanup preconditioners on host
  delete [] h_pc_lft;
  delete [] h_pc_rgt;

  // create backend
  switch(opts.type)
  {
    case kBackendPDHG:
      be = new be_pdhg(mat, d_precond_lft, d_precond_rgt,
                       prox_g, prox_hc, opts);
      break;

    default:
      return false;
  }

  // init backend
  be->initialize();

  return true;
}

void pdsolver::solve() {

  int i;
  for(i = 0; i < opts.max_iters; i++) {    
    be->do_iteration();

    bool is_converged = be->is_converged();
    
    if( (i == 0) ||
        (i == (opts.max_iters - 1)) ||
        (((i + 1) % opts.cb_iters) == 0) ||
        is_converged)
    {
      be->get_iterates(h_x, h_y);
      callback(i + 1, h_x, h_y, false);
    }

    if(is_converged)
      break;
  }
}

void pdsolver::release() {
  be->release();
  
  // @TODO really do this clean-up here? these are
  // allocated somewhere else. best thing would be to
  // use shared_pointers but that might be overkill
  for(int i = 0; i < prox_g.size(); ++i)
    delete prox_g[i];

  for(int i = 0; i < prox_hc.size(); ++i)
    delete prox_hc[i];

  cudaFree(d_precond_lft);
  cudaFree(d_precond_rgt);

  delete mat;
  delete be;

  delete [] h_x;
  delete [] h_y;
}
