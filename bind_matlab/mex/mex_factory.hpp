#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include "factory.hpp"

#include "prox/prox.hpp"
#include "prox/prox_moreau.hpp"
#include "prox/prox_zero.hpp"

#include "linop/linearoperator.hpp"
#include "linop/block.hpp"
#include "linop/block_zero.hpp"
#include "linop/block_sparse.hpp"

#include "backend/backend.hpp"
#include "backend/backend_pdhg.hpp"

#include "problem.hpp"
#include "solver.hpp"

// has to be included at end, otherwise 
// some compiler problems with std::printf 
#include "mex.h"
#include "mex_config.hpp"

class MexFactory 
{
public:
  static void Init();
     
  static Prox<real>* CreateProx(const mxArray *pm);
  static Block<real>* CreateBlock(const mxArray *pm);
  static Backend<real>* CreateBackend(const mxArray *pm);
  static Problem<real>* CreateProblem(const mxArray *pm);
  static typename Solver<real>::Options CreateSolverOptions(const mxArray *pm);

  static ProxMoreau<real>* CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data);   
  static ProxZero<real>* CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data);

  static BlockZero<real>* CreateBlockZero(size_t row, size_t col, const mxArray *data);
  static BlockSparse<real>* CreateBlockSparse(size_t row, size_t col, const mxArray *data);

  static BackendPDHG<real>* CreateBackendPDHG(const mxArray *data);

  static void PDHGStepsizeCallback(int iter, double res_primal, double res_dual, double& tau, double &sigma);
  static void SolverIntermCallback(int iter, const std::vector<real>& primal, const std::vector<real>& dual);

  static mxArray *PDHG_stepsize_cb_handle;
  static mxArray *Solver_interm_cb_handle;
};

#endif
