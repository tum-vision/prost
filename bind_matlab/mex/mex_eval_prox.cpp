#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "exception.hpp"
#include "mex_factory.hpp"

/// 
/// \brief Evaluates a proximal operator, this function mostly exists for
///        debugging purposes.
/// 
/// \param The proximal operator.
/// \param Prox argument.
/// \param Prox scalar step size.
/// \param Prox diagonal step size.
/// \returns Result of prox.
/// 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
  if(nrhs < 4)
    mexErrMsgTxt("At least four inputs required.");

  if(nlhs != 1)
    mexErrMsgTxt("One outputs required.");

  // check dimensions
  const mwSize *dims = mxGetDimensions(prhs[1]);
  size_t n = dims[0];
  if(dims[1] != 1)
    mexErrMsgTxt("Input to prox should be a vector!");

  
  // init prox
  std::shared_ptr<Prox<real>> prox;
  try 
  {
    prox = mex_factory::CreateProx(prhs[0]);
    prox->Initialize();
  } 
  catch(Exception& e) 
  {
    mexErrMsgTxt(e.what());
  }

   
  // read data from matlab
  double *arg = (double *)mxGetPr(prhs[1]);
  double *tau_diag = (double *)mxGetPr(prhs[3]);
  std::vector<real> h_result;
  std::vector<real> h_arg(arg, arg + n);
  std::vector<real> h_tau(tau_diag, tau_diag + n);
  real tau = (real) mxGetScalar(prhs[2]);

  // evaluate prox
  try
  {
    prox->Eval(h_result, h_arg, h_tau, tau);
  }
  catch(Exception& e)
  {
    mexErrMsgTxt(e.what());
  }

  // convert result back to MATLAB matrix and float -> double
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  std::copy(h_result.begin(), h_result.end(), (double *)mxGetPr(plhs[0]));
}
