#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mex_factory.hpp"
#include "linop/linop.hpp"
#include "util/util.hpp"

/**
 * @brief Evaluates a linear operator, this function mostly exists for
 *        debugging purposes.
 *                 
 * @param The linear operator L.
 * @param Right-hand side x.
 * @param Adjoint or normal evaluation?
 * @returns Result of y = L*x or y = L^T x if adjoint=true.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if(nrhs != 3)
    mexErrMsgTxt("Three inputs required.");

  if(nlhs != 1)
    mexErrMsgTxt("One output required.");

  // read input arguments
  LinearOperator<real> *linop = LinearOperatorFromMatlab(prhs[0]);
  double *rhs = mxGetPr(prhs[1]);
  bool transpose = (bool) mxGetScalar(prhs[2]);
  const mwSize *dims = mxGetDimensions(prhs[1]);
  size_t rhs_size = dims[0];
  size_t res_size;

  // sanity check
  if(dims[1] != 1)
    mexErrMsgTxt("RHS input to eval_linop should be a vector!");

  if(!linop->Init())
    mexErrMsgTxt("Failed to init linop!");

  res_size = transpose ? linop->ncols() : linop->nrows();

  if(transpose && (rhs_size != linop->nrows()))
    mexErrMsgTxt("rhs does not fit dimension of matrix!");
  else if(!transpose && (rhs_size != linop->ncols()))
    mexErrMsgTxt("rhs does not fit dimension of matrix!");

  //mexPrintf("Constructed linop of size %d x %d!\n", res_size, rhs_size);
  
  // allocate gpu arrays
  real *d_rhs;
  real *d_res;
  cudaMalloc((void **)&d_rhs, sizeof(real) * rhs_size);
  cudaMalloc((void **)&d_res, sizeof(real) * res_size);

  // convert double -> float if necessary
  real *h_rhs = new real[rhs_size];
  real *h_res = new real[res_size];
  for(size_t i = 0; i < rhs_size; i++) {
    h_rhs[i] = (real)rhs[i];
  }

  // fill prox arg and diag steps
  cudaMemcpy(d_rhs, h_rhs, sizeof(real) * rhs_size, cudaMemcpyHostToDevice);

  // evaluate prox
  Timer t;
  t.start();

  if(transpose)
    linop->EvalAdjoint(d_res, d_rhs);
  else
    linop->Eval(d_res, d_rhs);
  
  mexPrintf("LinOp took %f seconds.\n", t.get());
  
  // copy back result
  memset(h_res, 0, sizeof(real) * res_size);
  cudaMemcpy(h_res, d_res, sizeof(real) * res_size, cudaMemcpyDeviceToHost);

  // convert result back to MATLAB matrix and float -> double
  plhs[0] = mxCreateDoubleMatrix(res_size, 1, mxREAL);
  double *result_vals = mxGetPr(plhs[0]);
  for(size_t i = 0; i < res_size; i++)
    result_vals[i] = static_cast<double>(h_res[i]);

  // cleanup
  delete [] h_rhs;
  delete [] h_res;
  delete linop;

  cudaFree(d_rhs);
  cudaFree(d_res);
}
