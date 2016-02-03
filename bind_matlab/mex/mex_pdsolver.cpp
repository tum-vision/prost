#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "exception.hpp"
#include "mex_factory.hpp"

#ifdef __cplusplus
extern "C" bool utIsInterruptPending();
extern "C" void utSetInterruptPending(bool);
#else
extern bool utIsInterruptPending();
extern void utSetInterruptPending(bool);
#endif

bool
MexStoppingCallback() 
{
  if(utIsInterruptPending())
  {
    utSetInterruptPending(false);
    return true;
  }
  
  return false;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
  cudaDeviceReset();

  if(nrhs != 3)
    mexErrMsgTxt("Three inputs required. Usage: result = pdsolver(problem, backend, opts);");

  try 
  {
    mex_factory::Initialize();

    mexPrintf("Creating problem...\n");
    shared_ptr<Problem<real> > problem = mex_factory::CreateProblem(prhs[0]);
    mexPrintf("Creating backend...\n");
    shared_ptr<Backend<real> > backend = mex_factory::CreateBackend(prhs[1]);
    mexPrintf("Creating options...\n");
    typename Solver<real>::Options opts = mex_factory::CreateSolverOptions(prhs[2]);

    mexPrintf("Creating solver...\n");
    shared_ptr<Solver<real> > solver( new Solver<real>(problem, backend) );
    solver->SetOptions(opts);
    solver->SetIntermCallback(mex_factory::SolverIntermCallback);
    solver->SetStoppingCallback(MexStoppingCallback);

    mexPrintf("Initializing solver...\n");
    solver->Initialize();

    mexPrintf("Solving...\n");
    Solver<real>::ConvergenceResult result = solver->Solve();

    // Copy result back to MATLAB
    mxArray *mex_primal_sol = mxCreateDoubleMatrix(problem->ncols(), 1, mxREAL);
    mxArray *mex_dual_sol = mxCreateDoubleMatrix(problem->nrows(), 1, mxREAL);
    mxArray *result_string;

    switch(result)
    {
    case Solver<real>::ConvergenceResult::kConverged:
      result_string = mxCreateString("Converged.");
      break;

    case Solver<real>::ConvergenceResult::kStoppedMaxIters:
      result_string = mxCreateString("Reached maximum iterations.");
      break;

    case Solver<real>::ConvergenceResult::kStoppedUser:
      result_string = mxCreateString("Stopped by user.");
      break;
    }

    std::copy(solver->cur_dual_sol().begin(),
      solver->cur_dual_sol().end(),
      (double *)mxGetPr(mex_dual_sol));

    std::copy(solver->cur_primal_sol().begin(),
      solver->cur_primal_sol().end(),
      (double *)mxGetPr(mex_primal_sol));

    const char *fieldnames[3] = {
      "x",
      "y",
      "result"
    };

    plhs[0] = mxCreateStructMatrix(1, 1, 3, fieldnames);

    mxSetFieldByNumber(plhs[0], 0, 0, mex_primal_sol);
    mxSetFieldByNumber(plhs[0], 0, 1, mex_dual_sol);
    mxSetFieldByNumber(plhs[0], 0, 2, result_string);

    solver->Release();
  }
  catch(const Exception& e)
  {
    mexErrMsgTxt(e.what());
  }
}
