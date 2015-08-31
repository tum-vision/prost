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

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mex.h"
#include "prox_create.h"
#include "../pdsolver/pdsolver.h"

solver_options opts;
pdsolver solver;

void mex_pdsolver_callback(int it, real *x, real *y, bool is_converged) {
  // @todo call corresponding matlab function handle from mex
  std::cout << "mex_pdsolver_callback, it=" << it << std::endl;
}

void matrix_from_matlab(
    const mxArray *pm,
    real **val,
    int **ptr,
    int **ind,
    int &m,
    int &n,
    int &nnz)
{
  double *_val = mxGetPr(pm);
  mwIndex *_ind = mxGetIr(pm);
  mwIndex *_ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);
  
  m = dims[0];
  n = dims[1];
  nnz = _ptr[n];

  (*val) = new real[nnz];
  (*ptr) = new int[n + 1];
  (*ind) = new int[nnz];

  for(int i = 0; i < nnz; i++) {
    (*val)[i] = (real)_val[i];
    (*ind)[i] = (int)_ind[i];
  }
  
  for(int i = 0; i < n + 1; i++) {
    (*ptr)[i] = (int) _ptr[i];
  }
}

void opts_from_matlab(const mxArray *pm, solver_options& opts) {

  std::string be_name(mxArrayToString(mxGetField(pm, 0, "backend")));
  std::string pdhg_type(mxArrayToString(mxGetField(pm, 0, "pdhg_type")));

  std::transform(be_name.begin(),
                 be_name.end(),
                 be_name.begin(),
                 ::tolower);
  
  std::transform(pdhg_type.begin(),
                 pdhg_type.end(),
                 pdhg_type.begin(),
                 ::tolower);
  
  opts.max_iters = (int) mxGetScalar(mxGetField(pm, 0, "max_iters"));
  opts.cb_iters = (int) mxGetScalar(mxGetField(pm, 0, "cb_iters"));
  opts.tolerance = (real) mxGetScalar(mxGetField(pm, 0, "tolerance"));
  opts.gamma = (real) mxGetScalar(mxGetField(pm, 0, "gamma"));
  opts.alpha0 = (real) mxGetScalar(mxGetField(pm, 0, "alpha0"));
  opts.nu = (real) mxGetScalar(mxGetField(pm, 0, "nu"));
  opts.delta = (real) mxGetScalar(mxGetField(pm, 0, "delta"));
  opts.s = (real) mxGetScalar(mxGetField(pm, 0, "s"));

  if("pdhg" == be_name)
    opts.type = kBackendPDHG;

  if("alg1" == pdhg_type)
    opts.pdhg = kAlg1;
  else if("alg2" == pdhg_type)
    opts.pdhg = kAlg2;
  else if("adapt" == pdhg_type)
    opts.pdhg = kAdapt;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if(nrhs < 4)
    mexErrMsgIdAndTxt("pdsolver", "At least 3 inputs required.");

  if(!mxIsSparse(prhs[0]))
    mexErrMsgIdAndTxt("pdsolver", "The constraint matrix must be sparse.");

  if(nlhs != 2)
    mexErrMsgIdAndTxt("pdsolver", "Two outputs required.");
  
  // matrix K
  std::cout << "Building sparse constraint matrix...";
  int m, n, nnz;
  real *val;
  int *ptr, *ind;
  matrix_from_matlab(prhs[0], &val, &ptr, &ind, m, n, nnz);
  solver.set_matrix_csc(val, ptr, ind, m, n, nnz);
  std::cout << " done!" << std::endl;
  std::cout << "Problem size: n=" << n << ", m=" << m << ", nnz=" << nnz << std::endl;

  // output
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL); 
  plhs[1] = mxCreateDoubleMatrix(m, 1, mxREAL); 

  // options
  std::cout << "Reading options...";
  solver_options opts;
  opts_from_matlab(prhs[3], opts);
  std::cout << " done!" << std::endl;

  std::cout << "Options:" << std::endl;
  std::cout << " - Backend:";
  if(opts.type == kBackendPDHG)
  {
    std::cout << " PDHG,";

    switch(opts.pdhg)
    {
      case kAlg1:
        std::cout << " with constant steps (Alg. 1)." << std::endl;
        break;

      case kAlg2:
        std::cout << " accelerated version for strongly convex problems (Alg. 2). gamma = " << opts.gamma << std::endl;
        break;

      case kAdapt:
        std::cout << " adaptive step sizes. alpha0 = " << opts.alpha0;
        std::cout << ", nu = " << opts.nu;
        std::cout << ", delta = " << opts.delta;
        std::cout << ", s = " << opts.s << std::endl;
        break;
    }
  }
  std::cout << " - max_iters: " << opts.max_iters << std::endl;
  std::cout << " - cb_iters: " << opts.cb_iters << std::endl;
  std::cout << " - tolerance: " << opts.tolerance << std::endl;
  solver.set_backend(opts);

  // prox operators
  std::vector<prox *> prox_g;
  std::vector<prox *> prox_hc;

  if(!prox_from_matlab(prhs[1], prox_g))
    mexErrMsgIdAndTxt("pdsolver", "prox creation failed.");
  
  if(!prox_from_matlab(prhs[2], prox_hc))
    mexErrMsgIdAndTxt("pdsolver", "prox creation failed.");
  
  for(int i = 0; i < prox_g.size(); i++)
    solver.add_prox_g(prox_g[i]);
  
  for(int i = 0; i < prox_hc.size(); i++)
    solver.add_prox_hc(prox_hc[i]);

  // set iteration callback
  std::cout << "Set callback... ";
  solver.set_callback(&mex_pdsolver_callback);
  std::cout << " done!" << std::endl;

  // run solver
  std::cout << "Initializing... ";
  if(solver.initialize()) {

    std::cout << "done!" << std::endl << "Solving problem... ";
    solver.solve();
    std::cout << " done!" << std::endl;

    // convert result to double and write back to matlab
    real *primal = solver.get_primal_iterate();
    real *dual = solver.get_dual_iterate();

    double *matlab_primal = mxGetPr(plhs[0]);
    double *matlab_dual = mxGetPr(plhs[1]);
    
    for(int i = 0; i < n; ++i)
      matlab_primal[i] = (real)primal[i];
    
    for(int i = 0; i < m; ++i)
      matlab_dual[i] = (real)dual[i];
    
    solver.release();
  }
  else {
    mexErrMsgIdAndTxt("pdsolver", "initialization failed.");
  }

  
}
