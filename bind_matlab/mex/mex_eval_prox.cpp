#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mex_factory.hpp"
#include "util/util.hpp"

/**
 * @brief Evaluates a proximal operator, this function mostly exists for
 *        debugging purposes.
 *                 
 * @param The proximal operator.
 * @param Prox argument.
 * @param Prox scalar step size.
 * @param Prox diagonal step size.
 * @returns Result of prox.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if(nrhs < 4)
    mexErrMsgTxt("At least four inputs required.");

  if(nlhs != 1)
    mexErrMsgTxt("One outputs required.");

  // read input arguments
  Prox<real> *prox = ProxFromMatlab(prhs[0]);
  double *arg = mxGetPr(prhs[1]);
  real tau = (real) mxGetScalar(prhs[2]);
  double *tau_diag = mxGetPr(prhs[3]);
  const mwSize *dims = mxGetDimensions(prhs[1]);
  int n = dims[0];

  // sanity check
  if(dims[1] != 1)
    mexErrMsgTxt("Input to prox should be a vector!");

  if(!prox->Init())
    mexErrMsgTxt("Failed to init prox!");

  // allocate gpu arrays
  real *d_arg;
  real *d_result;
  real *d_tau;
  cudaMalloc((void **)&d_arg, sizeof(real) * n);
  cudaMalloc((void **)&d_result, sizeof(real) * n);
  cudaMalloc((void **)&d_tau, sizeof(real) * n);

  // convert double -> float if necessary
  real *h_arg = new real[n];
  real *h_result = new real[n];
  real *h_tau = new real[n];
  for(int i = 0; i < n; i++) {
    h_arg[i] = (real)arg[i];
    h_tau[i] = (real)tau_diag[i];
  }

  // fill prox arg and diag steps
  cudaMemcpy(d_arg, h_arg, sizeof(real) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tau, h_tau, sizeof(real) * n, cudaMemcpyHostToDevice);

  // evaluate prox
  Timer t;
  t.start();
  prox->Eval(d_arg, d_result, d_tau, tau);
  mexPrintf("prox took %f seconds.\n", t.get());

  // copy back result
  cudaMemcpy(h_result, d_result, sizeof(real) * n, cudaMemcpyDeviceToHost);

  // convert result back to MATLAB matrix and float -> double
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  double *result_vals = mxGetPr(plhs[0]);
  for(int i = 0; i < n; i++)
    result_vals[i] = (double) h_result[i];

  // cleanup
  delete [] h_arg;
  delete [] h_result;
  delete [] h_tau;
  delete prox;

  cudaFree(d_arg);
  cudaFree(d_result);
  cudaFree(d_tau);
}
