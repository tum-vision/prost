#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "factories_mex.hpp"

/**
 * @brief Calls the specified MATLAB callback function 
 */
void SolverCallback(int it, real *x, real *y, bool is_converged) {
  // TODO: call corresponding matlab function handle from mex
  //std::cout << "mex_pdsolver_callback, it=" << it << std::endl;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if(nrhs < 4)
    mexErrMsgIdAndTxt("pdsolver", "At least four inputs required.");

  if(!mxIsSparse(prhs[0]))
    mexErrMsgIdAndTxt("pdsolver", "The constraint matrix must be sparse.");

  if(nlhs != 2)
    mexErrMsgIdAndTxt("pdsolver", "Two outputs required.");
  
  Solver solver;
  SolverOptions opts;

  // set matrix
  SparseMatrix<real> *mat = MatrixFromMatlab(prhs[0]);
  solver.SetMatrix(mat);

  // set options
  OptsFromMatlab(prhs[3], opts);
  solver.SetOptions(opts);
  std::cout << opts.get_string() << std::endl;

  // set prox operators
  solver.SetProx_g(ProxListFromMatlab(prhs[1]));
  solver.SetProx_hc(ProxListFromMatlab(prhs[2]));

  // set callback
  solver.SetCallback(&SolverCallback);

  // init and run solver
  if(solver.Initialize()) {
    
    solver.Solve();

    real *primal = solver.get_primal_iterate();
    real *dual = solver.get_dual_iterate();

    int m, n;
    m = mat->nrows();
    n = mat->ncols();

    // allocate output
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(m, 1, mxREAL); 
    
    // convert result to double and write back to matlab
    double *matlab_primal = mxGetPr(plhs[0]);
    double *matlab_dual = mxGetPr(plhs[1]);
    
    for(int i = 0; i < n; ++i) matlab_primal[i] = (double)primal[i];    
    for(int i = 0; i < m; ++i) matlab_dual[i] = (double)dual[i];
    
    solver.Release();
  }
  else {
    mexErrMsgIdAndTxt("pdsolver", "Initialization failed!");
  }
}
