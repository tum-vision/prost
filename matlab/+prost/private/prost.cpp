#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "prost/common.hpp"
#include "prost/exception.hpp"
#include "factory.hpp"

using namespace mex; 
using namespace prost;

#ifdef __cplusplus
extern "C" bool utIsInterruptPending();
extern "C" void utSetInterruptPending(bool);
#else
extern bool utIsInterruptPending();
extern void utSetInterruptPending(bool);
#endif

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

static bool MexStoppingCallback() {
  if(utIsInterruptPending())
  {
    utSetInterruptPending(false);
    return true;
  }
  
  return false;
}

static void SolveProblem(MEX_ARGS) {

  std::shared_ptr<Problem<real> > problem = CreateProblem(prhs[0]);
  std::shared_ptr<Backend<real> > backend = CreateBackend(prhs[1]);
  typename Solver<real>::Options opts = CreateSolverOptions(prhs[2]);

  std::shared_ptr<Solver<real> > solver( new Solver<real>(problem, backend) );
  solver->SetOptions(opts);
  solver->SetIntermCallback(SolverIntermCallback);
  solver->SetStoppingCallback(MexStoppingCallback);

  solver->Initialize();

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

static void EvalLinOp(MEX_ARGS) {
  // TODO
}

static void EvalProx(MEX_ARGS) {
  // TODO
}

struct command_registry {
  string cmd;
  function<void(MEX_ARGS)> fn_handle;
};

static command_registry cmd_reg[] = {
  { "solve_problem", SolveProblem },
  { "eval_linop",    EvalLinOp    },
  { "eval_prox",     EvalProx     },
  // The end.
  { "END",           nullptr      },
};

void mexFunction(MEX_ARGS) {
  mexLock();
  
  if(nrhs == 0)
    mexErrMsgTxt("Usage: prost_(command, arg1, arg2, ...);");

  char *cmd = mxArrayToString(prhs[0]);
  bool executed = false;
  
  try 
  {
    for(size_t i = 0; cmd_reg[i].fn_handle != nullptr; i++) {
      if(cmd_reg[i].cmd.compare(cmd) == 0) {
        cmd_reg[i].fn_handle(nlhs, plhs, nrhs - 1, prhs + 1);
        executed = true;
        break;
      }
    }

    if(!executed) {
      stringstream msg;
      msg << "Unknown command '" << cmd << "'.";
      throw Exception(msg.str());
    }
  }
  catch(const Exception& e)
  {
    mexErrMsgTxt(e.what());
  }
}
