#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mex_factory.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
  if(nrhs < 2)
  {
    mexErrMsgTxt("At least two inputs required.\nUsage: result = pdsolver(problem, backend, [opts]);");
  }

  try 
  {
    // build problem specification
    // read blocks and build linear operator
    // read prox
    // read preconditioners / compute preconditioners if none available
    
    // build backend
    // read options etc

    // instantiate solver class

  }
  catch(const Exception& e)
  {
  }
}
