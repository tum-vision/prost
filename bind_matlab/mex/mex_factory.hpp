#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include "factory.hpp"

#include "prox/prox.hpp"
#include "prox/prox_moreau.hpp"
#include "prox/prox_zero.hpp"

#include "linop/linearoperator.hpp"
#include "linop/block.hpp"
#include "linop/block_zero.hpp"

#include "backend/backend.hpp"
#include "backend/backend_pdhg.hpp"

#include "exception.hpp"
#include "problem.hpp"
#include "solver.hpp"

#include <algorithm>
#include <memory>
#include <string>

// has to be included at end, otherwise 
// some compiler problems with std::printf 
#include "mex.h"
#include "mex_config.hpp"

typedef Factory<Prox<real>, size_t, size_t, bool, const mxArray*> ProxFactory;
typedef Factory<Block<real>, size_t, size_t, const mxArray*> BlockFactory;
typedef Factory<Backend<real>, const mxArray*> BackendFactory;

class MexFactory 
{
public:
  static void Init() 
  {
    // prox operators
    ProxFactory::GetInstance()->Register<ProxMoreau<real>* >("moreau", CreateProxMoreau);
    ProxFactory::GetInstance()->Register<ProxZero<real>* >("zero", CreateProxZero);

    // blocks
    BlockFactory::GetInstance()->Register<BlockZero<real>* >("zero", CreateBlockZero);

    // backends
    BackendFactory::GetInstance()->Register<BackendPDHG<real>* >("pdhg", CreateBackendPDHG);
  }
  
  static ProxMoreau<real>* CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
  {
    return new ProxMoreau<real>(std::unique_ptr<Prox<real> >(CreateProx(mxGetCell(data, 0))));
  }
    
  static ProxZero<real>* CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
  {
    return new ProxZero<real>(idx, size);
  }

  static BlockZero<real>* CreateBlockZero(size_t row, size_t col, const mxArray *data)
  {
    size_t nrows = (size_t) mxGetScalar(mxGetCell(data, 0));
    size_t ncols = (size_t) mxGetScalar(mxGetCell(data, 1));

    return new BlockZero<real>(row, col, nrows, ncols);
  }

  static BackendPDHG<real>* CreateBackendPDHG(const mxArray *data)
  {
    typename BackendPDHG<real>::Options opts;

    // read options from data
    opts.tau0 = (real) mxGetScalar(mxGetField(data, 0, "tau0"));
    opts.sigma0 = (real) mxGetScalar(mxGetField(data, 0, "sigma0"));
    opts.solve_dual_problem = (real) mxGetScalar(mxGetField(data, 0, "solve_dual"));
    opts.residual_iter = (int) mxGetScalar(mxGetField(data, 0, "residual_iter"));
    opts.scale_steps_operator = (bool) mxGetScalar(mxGetField(data, 0, "scale_steps_operator"));;
    opts.alg2_gamma = (real) mxGetScalar(mxGetField(data, 0, "alg2_gamma"));
    opts.arg_alpha0 = (real) mxGetScalar(mxGetField(data, 0, "arg_alpha0"));
    opts.arg_nu = (real) mxGetScalar(mxGetField(data, 0, "arg_nu"));
    opts.arg_delta = (real) mxGetScalar(mxGetField(data, 0, "arg_delta"));
    opts.arb_delta = (real) mxGetScalar(mxGetField(data, 0, "arb_delta"));
    opts.arb_tau = (real) mxGetScalar(mxGetField(data, 0, "arb_tau"));

    std::string stepsize_variant(mxArrayToString(mxGetField(data, 0, "stepsize")));
    std::transform(stepsize_variant.begin(), stepsize_variant.end(), stepsize_variant.begin(), ::tolower);

    // TODO: do this nicer with C++11?
    if(stepsize_variant == "alg1")
      opts.stepsize_variant = BackendPDHG<real>::StepsizeVariant::kPDHGStepsAlg1;
    else if(stepsize_variant == "alg2")
      opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsAlg2;
    else if(stepsize_variant == "goldstein")
      opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsResidualGoldstein;
    else if(stepsize_variant == "boyd")
      opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsResidualBoyd;
    else if(stepsize_variant == "callback")
      opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsCallback;
    else
      throw new Exception("Couldn't recognize step-size variant.");

    PDHG_stepsize_cb_handle = mxGetField(data, 0, "stepsize_callback");

    BackendPDHG<real> *backend = new BackendPDHG<real>(opts);
    backend->SetStepsizeCallback(PDHGStepsizeCallback);

    return backend;
  }
    
  static Prox<real>* CreateProx(const mxArray *pm) 
  {
    std::string name(mxArrayToString(mxGetCell(pm, 0)));
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    size_t idx = (size_t) mxGetScalar(mxGetCell(pm, 1));
    size_t size = (size_t) mxGetScalar(mxGetCell(pm, 2));
    bool diagsteps = (bool) mxGetScalar(mxGetCell(pm, 3));
    mxArray *data = mxGetCell(pm, 4);
        
    return ProxFactory::GetInstance()->Create(name, idx, size, diagsteps, data);
  }

  static Block<real>* CreateBlock(const mxArray *pm)
  {
    std::string name(mxArrayToString(mxGetCell(pm, 0)));
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    size_t row = (size_t) mxGetScalar(mxGetCell(pm, 1));
    size_t col = (size_t) mxGetScalar(mxGetCell(pm, 2));
    mxArray *data = mxGetCell(pm, 3);

    return BlockFactory::GetInstance()->Create(name, row, col, data);
  }
    
  static Backend<real>* CreateBackend(const mxArray *pm)
  {
    std::string name(mxArrayToString(mxGetCell(pm, 0)));
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    return BackendFactory::GetInstance()->Create(name, mxGetCell(pm, 1));
  }

  static Problem<real>* CreateProblem(const mxArray *pm)
  {
    Problem<real> *prob = new Problem<real>;

    // add blocks
    mxArray *cell_linop = mxGetField(pm, 0, "linop");
    const mwSize *dims_linop = mxGetDimensions(cell_linop);

    for(mwSize i = 0; i < dims_linop[0]; i++)
      prob->AddBlock( CreateBlock(mxGetCell(cell_linop, i)) );

    // add proxs
    mxArray *cell_prox_g = mxGetField(pm, 0, "prox_g");
    mxArray *cell_prox_f = mxGetField(pm, 0, "prox_f");
    mxArray *cell_prox_gstar = mxGetField(pm, 0, "prox_gstar");
    mxArray *cell_prox_fstar = mxGetField(pm, 0, "prox_fstar");

    const mwSize *dims_prox_g = mxGetDimensions(cell_prox_g);
    const mwSize *dims_prox_f = mxGetDimensions(cell_prox_f);
    const mwSize *dims_prox_gstar = mxGetDimensions(cell_prox_gstar);
    const mwSize *dims_prox_fstar = mxGetDimensions(cell_prox_fstar);

    for(mwSize i = 0; i < dims_prox_g[0]; i++) 
      prob->AddProx_g( CreateProx(mxGetCell(cell_prox_g, i)) );

    for(mwSize i = 0; i < dims_prox_f[0]; i++) 
      prob->AddProx_f( CreateProx(mxGetCell(cell_prox_f, i)) );

    for(mwSize i = 0; i < dims_prox_gstar[0]; i++) 
      prob->AddProx_gstar( CreateProx(mxGetCell(cell_prox_gstar, i)) );

    for(mwSize i = 0; i < dims_prox_fstar[0]; i++) 
      prob->AddProx_fstar( CreateProx(mxGetCell(cell_prox_fstar, i)) );

    // set scaling
    std::string scaling(mxArrayToString(mxGetField(pm, 0, "scaling")));
    std::transform(scaling.begin(), scaling.end(), scaling.begin(), ::tolower);

    if(scaling == "alpha")
      prob->SetScalingAlpha( (real) mxGetScalar(mxGetField(pm, 0, "scaling_alpha")) );
    else if(scaling == "identity")
      prob->SetScalingIdentity();
    else if(scaling == "custom")
    {
      std::vector<real> left, right;

      const mwSize *dim_scaling_left = mxGetDimensions(mxGetField(pm, 0, "scaling_left"));
      const mwSize *dim_scaling_right = mxGetDimensions(mxGetField(pm, 0, "scaling_right"));

      double *vals_scaling_left = mxGetPr(mxGetField(pm, 0, "scaling_left"));
      double *vals_scaling_right = mxGetPr(mxGetField(pm, 0, "scaling_right"));

      for(size_t i = 0; i < dim_scaling_left[0]; ++i)
        left.push_back( vals_scaling_left[i] );

      for(size_t i = 0; i < dim_scaling_right[0]; ++i)
        right.push_back( vals_scaling_right[i] );

      prob->SetScalingCustom(left, right);
    }
    else
      throw new Exception("Problem scaling variant not recognized.");

    return prob;
  }

  static typename Solver<real>::Options CreateSolverOptions(const mxArray *pm)
  {
    typename Solver<real>::Options opts;

    opts.tol_rel_primal = (real) mxGetScalar(mxGetField(pm, 0, "tol_rel_primal"));
    opts.tol_rel_dual = (real) mxGetScalar(mxGetField(pm, 0, "tol_rel_dual"));
    opts.tol_abs_primal = (real) mxGetScalar(mxGetField(pm, 0, "tol_abs_primal"));
    opts.tol_abs_dual = (real) mxGetScalar(mxGetField(pm, 0, "tol_abs_dual"));
    opts.max_iters = (int) mxGetScalar(mxGetField(pm, 0, "max_iters"));
    opts.num_cback_calls = (int) mxGetScalar(mxGetField(pm, 0, "num_cback_calls"));
    opts.verbose = (bool) mxGetScalar(mxGetField(pm, 0, "verbose"));

    Solver_interm_cb_handle = mxGetField(pm, 0, "interm_cb");

    return opts;
  }

  static mxArray *PDHG_stepsize_cb_handle;
  static mxArray *Solver_interm_cb_handle;

  static void PDHGStepsizeCallback(int iter, double res_primal, double res_dual, double& tau, double &sigma)
  {
    // TODO: Call MATLAB function PDHG_stepsize_cb_handle
  }

  static void SolverIntermCallback(int iter, const std::vector<real>& primal, const std::vector<real>& dual)
  {
    // TODO: Call MATLAB function Solver_interm_cb_handle
  }

};

mxArray* MexFactory::PDHG_stepsize_cb_handle = nullptr;
mxArray* MexFactory::Solver_interm_cb_handle = nullptr;

#endif
