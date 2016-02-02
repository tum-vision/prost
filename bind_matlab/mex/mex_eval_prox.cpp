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

  MexFactory::Init();
  
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
    mexErrMsgTxt(e.what());
  }
  
  // convert double -> float if necessary
  std::vector<real> h_arg(n);
  std::vector<real> h_result(n);
  std::vector<real> h_tau(n);
  for(int i = 0; i < n; i++) {
    h_arg[i] = (real)arg[i];
    h_tau[i] = (real)tau_diag[i];
  }

  // evaluate prox
  Timer t;
  t.start();

  prox->Eval(h_arg, h_result, h_tau, tau);

  mexPrintf("Evaluation of prox took %f seconds.\n", t.get());


  // convert result back to MATLAB matrix and float -> double
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  double *result_vals = mxGetPr(plhs[0]);
  for(int i = 0; i < n; i++)
      result_vals[i] = (double) h_result[i];
}
