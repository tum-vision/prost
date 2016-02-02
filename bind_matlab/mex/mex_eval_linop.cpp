#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "exception.hpp"
#include "mex_factory.hpp"

/// \brief Evaluates a linear operator, this function mostly exists for
///        debugging purposes.
/// 
/// \param The linear operator L.
/// \param Right-hand side x.
/// \param Adjoint or normal evaluation?
/// \returns Result of y = L*x or y = L^T x if adjoint=true, as well as
///          an array rowsum and colsum (for alpha = 1).

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
  if(nrhs != 3)
    mexErrMsgTxt("Three inputs required");

  if(nlhs != 3)
    mexErrMsgTxt("Three outputs (result, rowsum, colsum) required.");

  MexFactory::Init();
  
  // read input arguments
  std::shared_ptr<LinearOperator<real> > linop(new LinearOperator<real>());

  const mxArray *cell_linop = prhs[0];
  const mwSize *dims_linop = mxGetDimensions(cell_linop);
  for(mwSize i = 0; i < dims_linop[0]; i++)
    linop->AddBlock( std::shared_ptr<Block<real> >(MexFactory::CreateBlock(mxGetCell(cell_linop, i))) );

  std::vector<real> rhs;
  bool transpose = static_cast<bool>(mxGetScalar(prhs[2]));
  const mwSize *dims = mxGetDimensions(prhs[1]);

  if(dims[1] != 1)
    mexErrMsgTxt("Right-hand side input to eval_linop should be a n-times-1 vector!");

  try 
  {
    linop->Initialize();
  } 
  catch(const Exception& e) 
  {
    std::ostringstream ss;
    ss << "Failed to initialize LinearOperator: " << e.what() << "\n";
    mexPrintf(ss.str().c_str());
  }

  double *matlab_rhs = (double *)mxGetPr(prhs[1]);

  if(transpose)
    rhs = std::vector<real>( matlab_rhs, matlab_rhs + linop->nrows());
  else
    rhs = std::vector<real>( matlab_rhs, matlab_rhs + linop->ncols());

  std::vector<real> res;
  try
  {
    if(transpose)
      linop->EvalAdjoint(res, rhs);
    else 
      linop->Eval(res, rhs);
  }
  catch(const Exception& e)
  {
    std::ostringstream ss;
    ss << "Failed to evaluate LinearOperator: " << e.what() << "\n";
    mexPrintf(ss.str().c_str());
  }

  plhs[0] = mxCreateDoubleMatrix(transpose ? linop->ncols() : linop->nrows(), 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(linop->nrows(), 1, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(linop->ncols(), 1, mxREAL);

  std::copy(res.begin(), res.end(), (double *)mxGetPr(plhs[0]));

  std::vector<double> rowsum(linop->nrows());
  std::vector<double> colsum(linop->ncols());

  for(size_t row = 0; row < linop->nrows(); row++)
    rowsum[row] = linop->row_sum(row, 1);

  for(size_t col = 0; col < linop->ncols(); col++)
    colsum[col] = linop->col_sum(col, 1);

  std::copy(rowsum.begin(), rowsum.end(), (double *)mxGetPr(plhs[1]));
  std::copy(colsum.begin(), colsum.end(), (double *)mxGetPr(plhs[2]));
}
