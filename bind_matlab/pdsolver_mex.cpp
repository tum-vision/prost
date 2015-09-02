#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "factory_mex.hpp"

int m, n;
mxArray *cb_rhs[4];

/**
 * @brief Calls the specified MATLAB callback function 
 */
void SolverCallback(int it, real *x, real *y, bool is_converged) {
  cb_rhs[1] = mxCreateDoubleScalar(it);
  cb_rhs[2] = mxCreateDoubleMatrix(n, 1, mxREAL); 
  cb_rhs[3] = mxCreateDoubleMatrix(m, 1, mxREAL);

  double *matlab_primal = mxGetPr(cb_rhs[2]);
  double *matlab_dual = mxGetPr(cb_rhs[3]);
    
  for(int i = 0; i < n; ++i) matlab_primal[i] = (double)x[i];    
  for(int i = 0; i < m; ++i) matlab_dual[i] = (double)y[i];
  
  mexCallMATLAB(0, NULL, 4, cb_rhs, "feval");
}

void ProxListFromMatlab(const mxArray *pm, std::vector<Prox *>& proxs) {
  const mwSize *dims = mxGetDimensions(pm);
  int num_proxs = dims[0];

  for(int i = 0; i < num_proxs; i++) {
    mxArray *prox_cell = mxGetCell(pm, i);
    Prox *prox = ProxFromMatlab(prox_cell);

    if(0 == prox)
      mexErrMsgTxt("Error creating the prox.");
    else
      proxs.push_back(prox);
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if(nrhs < 5)
    mexErrMsgTxt("At least five inputs required.");

  if(!mxIsSparse(prhs[0]))
    mexErrMsgTxt("The constraint matrix must be sparse.");

  if(nlhs != 2)
    mexErrMsgTxt("Two outputs required.");
  
  Solver solver;
  SolverOptions opts;

  // set matrix
  SparseMatrix<real> *mat = MatrixFromMatlab(prhs[0]);
  solver.SetMatrix(mat);
  m = mat->nrows();
  n = mat->ncols();

  // set options
  SolverOptionsFromMatlab(prhs[3], opts);
  solver.SetOptions(opts);
  //std::cout << opts.get_string() << std::endl;
  mexPrintf("%s", opts.get_string().c_str());

  // set prox operators, ownership gets transfered to solver
  std::vector<Prox *> prox_g, prox_hc;
  ProxListFromMatlab(prhs[1], prox_g);
  ProxListFromMatlab(prhs[2], prox_hc);
  solver.SetProx_g(prox_g);
  solver.SetProx_hc(prox_hc);

  // set callback
  solver.SetCallback(&SolverCallback);
  cb_rhs[0] = const_cast<mxArray *>(prhs[4]);
  
  // init and run solver
  if(solver.Initialize()) {
    mexPrintf("Initialized solver successfully!\n");

    size_t gpu_mem_avail, gpu_mem_required;
    solver.gpu_mem_amount(gpu_mem_required, gpu_mem_avail);

    mexPrintf("%.3fMBytes of GPU memory required (%.3fMBytes available).\n",
              (double)gpu_mem_required / (1024. * 1024.),
              (double)gpu_mem_avail / (1024. * 1024.));

    solver.Solve();

    real *primal = solver.primal_iterate();
    real *dual = solver.dual_iterate();

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
