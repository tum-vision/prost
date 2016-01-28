#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "mex_factory.hpp"

#ifdef __cplusplus
    extern "C" bool utIsInterruptPending();
    extern "C" void utSetInterruptPending(bool);
#else
    extern bool utIsInterruptPending();
    extern void utSetInterruptPending(bool);
#endif

bool MexStoppingCallback() 
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
    MexFactory::Init();

    shared_ptr<Problem<real> > problem( MexFactory::CreateProblem(prhs[0]) );
    shared_ptr<Backend<real> > backend( MexFactory::CreateBackend(prhs[1]) );
    typename Solver<real>::Options opts = MexFactory::CreateSolverOptions(prhs[2]);

    shared_ptr<Solver<real> > solver( new Solver<real>(problem, backend) );
    solver->SetOptions(opts);
    solver->SetIntermCallback(MexFactory::SolverIntermCallback);
    solver->SetStoppingCallback(MexStoppingCallback);

    solver->Initialize();
    Solver<real>::ConvergenceResult result = solver->Solve();

    // Copy result back to MATLAB
    mxArray *mx_primal_sol = mxCreateDoubleMatrix(problem->ncols(), 1, mxREAL);
    mxArray *mx_dual_sol = mxCreateDoubleMatrix(problem->nrows(), 1, mxREAL);
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

    double *primal_sol = mxGetPr(mx_primal_sol);
    double *dual_sol = mxGetPr(mx_dual_sol);

    for(size_t i = 0; i < problem->nrows(); i++)
      dual_sol[i] = static_cast<double>(solver->cur_dual_sol()[i]);

    for(size_t i = 0; i < problem->ncols(); i++)
      primal_sol[i] = static_cast<double>(solver->cur_primal_sol()[i]);     

    const char *fieldnames[3] = {
      "x",
      "y",
      "result"
    };

    plhs[0] = mxCreateStructMatrix(1, 1, 3, fieldnames);

    mxSetFieldByNumber(plhs[0], 0, 0, mx_primal_sol);
    mxSetFieldByNumber(plhs[0], 0, 1, mx_dual_sol);
    mxSetFieldByNumber(plhs[0], 0, 2, result_string);

    solver->Release();
  }
  catch(const Exception& e)
  {
    mexErrMsgTxt(e.what());
  }
}
