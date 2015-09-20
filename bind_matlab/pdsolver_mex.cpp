#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "factory_mex.hpp"

// global variables for callback
int mat_nrows, mat_ncols;
mxArray *cb_func_handle = NULL;

/**
 * @brief Calls the specified MATLAB callback function 
 */
void SolverCallback(int it, real *x, real *y, bool is_converged) {
  mxArray *cb_rhs[4];
  cb_rhs[0] = cb_func_handle;
  cb_rhs[1] = mxCreateDoubleScalar(it);
  cb_rhs[2] = mxCreateDoubleMatrix(mat_ncols, 1, mxREAL); 
  cb_rhs[3] = mxCreateDoubleMatrix(mat_nrows, 1, mxREAL);

  double *prim = mxGetPr(cb_rhs[2]);
  double *dual = mxGetPr(cb_rhs[3]);
    
  for(int i = 0; i < mat_ncols; ++i) prim[i] = (double) x[i];    
  for(int i = 0; i < mat_nrows; ++i) dual[i] = (double) y[i];
  
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

__host__ void cleanUp() 
{
  cudaDeviceReset();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  cudaDeviceReset();
  mexAtExit(cleanUp);
  
  if(nrhs < 4)
    mexErrMsgTxt("At least four inputs required.");

  if(!mxIsSparse(prhs[0]))
    mexErrMsgTxt("The constraint matrix must be sparse.");

  if(nlhs != 2)
    mexErrMsgTxt("Two outputs required.");

  /*
  size_t gpu_mem_free, gpu_mem_avail, gpu_mem_required;
  cudaMemGetInfo(&gpu_mem_free, &gpu_mem_avail);
  mexPrintf("Free GPU memory: %.3f (%.3f available)\n", gpu_mem_free / (1024. * 1024.), gpu_mem_avail / (1024. * 1024.));
  */
  
  Solver solver;
  SolverOptions opts;

  // set matrix
  SparseMatrix<real> *mat = MatrixFromMatlab(prhs[0]);
  solver.SetMatrix(mat);
  mat_nrows = mat->nrows();
  mat_ncols = mat->ncols();

  // set options
  SolverOptionsFromMatlab(prhs[3], opts, &cb_func_handle);
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
  
  // init and run solver
  if(solver.Initialize()) {
    size_t gpu_mem_required, gpu_mem_avail, gpu_mem_free;
    
    solver.gpu_mem_amount(gpu_mem_required, gpu_mem_avail, gpu_mem_free);
    mexPrintf("%.3fMBytes of GPU memory required (%.3fMBytes available, %.3fMBytes free!)\n",
              (double)gpu_mem_required / (1024. * 1024.),
              (double)gpu_mem_avail / (1024. * 1024.),
              (double)gpu_mem_free / (1024. * 1024.));

    if(gpu_mem_required > gpu_mem_avail) {
      mexErrMsgTxt("Out of memory!");
    }
    
    mexPrintf("Initialized solver successfully!\n");

    solver.Solve();

    real *prim = solver.primal_iterate();
    real *dual = solver.dual_iterate();

    // allocate output
    plhs[0] = mxCreateDoubleMatrix(mat_ncols, 1, mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(mat_nrows, 1, mxREAL); 
    
    // convert result to double and write back to matlab
    double *mat_prim = mxGetPr(plhs[0]);
    double *mat_dual = mxGetPr(plhs[1]);
    
    for(int i = 0; i < mat_ncols; ++i) mat_prim[i] = (double)prim[i];    
    for(int i = 0; i < mat_nrows; ++i) mat_dual[i] = (double)dual[i];
    
    solver.Release();
  }
  else {
    mexErrMsgTxt("Initialization failed!");
  }
}
