#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mex_factory.hpp"
#include "util/util.hpp"
#include "config.hpp"
#include <thrust/device_vector.h>

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
  unique_ptr<prox::Prox<real>> prox = move(MexFactory::CreateProx(prhs[0]));
  double *arg = mxGetPr(prhs[1]);
  real tau = (real) mxGetScalar(prhs[2]);
  double *tau_diag = mxGetPr(prhs[3]);
  const mwSize *dims = mxGetDimensions(prhs[1]);
  int n = dims[0];

  // sanity check
  if(dims[1] != 1)
    mexErrMsgTxt("Input to prox should be a vector!");

  try {
    prox->Init();
  } catch(exception& e) {
    mexErrMsgTxt("Failed to init prox!");
  }
  
  std::cout << sizeof(real)<<std::endl;;
  // allocate gpu arrays
  thrust::device_vector<real> d_arg(n);
  thrust::device_vector<real> d_result(n);
  thrust::device_vector<real> d_tau(n);

  // convert double -> float if necessary
  thrust::host_vector<real> h_arg(n);
  thrust::host_vector<real> h_result(n);
  thrust::host_vector<real> h_tau(n);
  for(int i = 0; i < n; i++) {
    h_arg[i] = (real)arg[i];
    h_tau[i] = (real)tau_diag[i];
  }

  d_arg = h_arg;
  d_tau = h_tau;

  // evaluate prox
  Timer t;
  t.start();
  prox->Eval(d_arg, d_result, d_tau, tau);
  mexPrintf("prox took %f seconds.\n", t.get());

  // copy back result
  h_result = d_result;

  // convert result back to MATLAB matrix and float -> double
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  double *result_vals = mxGetPr(plhs[0]);
  for(int i = 0; i < n; i++)
      result_vals[i] = (double) h_result[i];
}
