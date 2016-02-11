#include "mex_factory.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "prox/elemop/function_1d.hpp"
#include "exception.hpp"

namespace mex_factory 
{

mxArray *PDHG_stepsize_cb_handle = nullptr;
mxArray *Solver_interm_cb_handle = nullptr;

ProxRegistry prox_reg[] = 
{
  { "moreau",                         CreateProxMoreau                                       },
  { "zero",                           CreateProxZero                                         },
  { "elem_operation:simplex",         CreateProxElemOperationSimplex                         },
  { "elem_operation:1d:zero",         CreateProxElemOperation1D<Function1DZero<real>>        },
  { "elem_operation:1d:abs",          CreateProxElemOperation1D<Function1DAbs<real>>         },
  { "elem_operation:1d:square",       CreateProxElemOperation1D<Function1DSquare<real>>      },
  { "elem_operation:1d:ind_leq0",     CreateProxElemOperation1D<Function1DIndLeq0<real>>     },
  { "elem_operation:1d:ind_geq0",     CreateProxElemOperation1D<Function1DIndGeq0<real>>     },
  { "elem_operation:1d:ind_eq0",      CreateProxElemOperation1D<Function1DIndEq0<real>>      },
  { "elem_operation:1d:ind_box01",    CreateProxElemOperation1D<Function1DIndBox01<real>>    },
  { "elem_operation:1d:max_pos0",     CreateProxElemOperation1D<Function1DMaxPos0<real>>     },
  { "elem_operation:1d:l0",           CreateProxElemOperation1D<Function1DL0<real>>          },
  { "elem_operation:1d:huber",        CreateProxElemOperation1D<Function1DHuber<real>>       },
  { "elem_operation:norm2:zero",      CreateProxElemOperationNorm2<Function1DZero<real>>     },
  { "elem_operation:norm2:abs",       CreateProxElemOperationNorm2<Function1DAbs<real>>      },
  { "elem_operation:norm2:square",    CreateProxElemOperationNorm2<Function1DSquare<real>>   },
  { "elem_operation:norm2:ind_leq0",  CreateProxElemOperationNorm2<Function1DIndLeq0<real>>  },
  { "elem_operation:norm2:ind_geq0",  CreateProxElemOperationNorm2<Function1DIndGeq0<real>>  },
  { "elem_operation:norm2:ind_eq0",   CreateProxElemOperationNorm2<Function1DIndEq0<real>>   },
  { "elem_operation:norm2:ind_box01", CreateProxElemOperationNorm2<Function1DIndBox01<real>> },
  { "elem_operation:norm2:max_pos0",  CreateProxElemOperationNorm2<Function1DMaxPos0<real>>  },
  { "elem_operation:norm2:l0",        CreateProxElemOperationNorm2<Function1DL0<real>>       },
  { "elem_operation:norm2:huber",     CreateProxElemOperationNorm2<Function1DHuber<real>>    },
  { "epi_quadratic_fun",              CreateProjEpiQuadraticFun                              },
  
  // The end.
  { "END",                            nullptr                                                },
};

BlockRegistry block_reg[] = 
{
  { "zero",   CreateBlockZero   },
  { "sparse", CreateBlockSparse },

  // The end.
  { "END",    nullptr           },
};

BackendRegistry backend_reg[] = 
{
  { "pdhg", CreateBackendPDHG },

  // The end.
  { "END",  nullptr           },
};

void
PDHGStepsizeCallback(int iter, double res_primal, double res_dual, double& tau, double &sigma)
{
  // TODO: Call MATLAB function PDHG_stepsize_cb_handle
}

void
SolverIntermCallback(int iter, const std::vector<real>& primal, const std::vector<real>& dual)
{
  // TODO: Call MATLAB function Solver_interm_cb_handle
}

ProxMoreau<real>* 
CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  return new ProxMoreau<real>(CreateProx(mxGetCell(data, 0)));
}
    
ProxZero<real>* 
CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  return new ProxZero<real>(idx, size);
}

BlockZero<real>* 
CreateBlockZero(size_t row, size_t col, const mxArray *data)
{
  size_t nrows = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t ncols = (size_t) mxGetScalar(mxGetCell(data, 1));
  
  return new BlockZero<real>(row, col, nrows, ncols);
}

template<size_t COEFFS_COUNT>
void GetCoefficients(
  std::array<std::vector<real>, COEFFS_COUNT>& coeffs, 
  const mxArray *coeffs_mx, 
  size_t count) 
{
  for(size_t i = 0; i < COEFFS_COUNT; i++) 
  {    
    const mwSize *dims = mxGetDimensions(mxGetCell(coeffs_mx, i));
    
    if(dims[0] != 1 && dims[0] != count) 
      throw Exception("Prox: Dimension of coefficients has to be equal to 1 or count\n");
        
    double *val = mxGetPr(mxGetCell(coeffs_mx, i));
    coeffs[i] = std::vector<real>(val, val + dims[0]);
  }
}

template<class FUN_1D> 
ProxElemOperation<real, ElemOperation1D<real, FUN_1D> >*
CreateProxElemOperation1D(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  std::array<std::vector<real>, 7> coeffs;
  GetCoefficients<7>(coeffs, mxGetCell(data, 3), size);

  return new ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>(
    idx, count, dim, interleaved, diagsteps, coeffs);   
}

template<class FUN_1D>
ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D> >* 
CreateProxElemOperationNorm2(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  std::array<std::vector<real>, 7> coeffs;
  GetCoefficients<7>(coeffs, mxGetCell(data, 3), count);
  
  return new ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D>>(
    idx, count, dim, interleaved, diagsteps, coeffs);   
}

ProxElemOperation<real, ElemOperationSimplex<real> >* 
CreateProxElemOperationSimplex(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  return new ProxElemOperation<real, ElemOperationSimplex<real> >(idx, count, dim, interleaved, diagsteps);   
}

ProjEpiQuadraticFun<real>* 
CreateProjEpiQuadraticFun(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));
 
  const mxArray *coeffs =  mxGetCell(data, 3);
  
  const mwSize *dims = mxGetDimensions(mxGetCell(coeffs, 0));
    
  if(dims[0] != 1 && dims[0] != count)
    throw Exception("Prox: Dimension of coefficient a has to be equal to 1 or count\n");
        
  double *val = mxGetPr(mxGetCell(coeffs, 0));
  std::vector<real> a(val, val + dims[0]);
  
  dims = mxGetDimensions(mxGetCell(coeffs, 1));
    
  if(dims[0] != count*(dim-1) && dims[0] != dim-1)
    throw Exception("Prox: Dimension of coefficient b has to be equal to dim-1 or count*(dim-1)\n");
        
  val = mxGetPr(mxGetCell(coeffs, 1));
  std::vector<real> b(val, val + dims[0]);
  
  dims = mxGetDimensions(mxGetCell(coeffs, 2));
    
  if(dims[0] != 1 && dims[0] != count)
    throw Exception("Prox: Dimension of coefficient c has to be equal to 1 or count\n");
        
  val = mxGetPr(mxGetCell(coeffs, 2));
  std::vector<real> c(val, val + dims[0]);
  
  return new ProjEpiQuadraticFun<real>(idx, count, dim, interleaved, diagsteps, a, b, c);   
}

BlockSparse<real>*
CreateBlockSparse(size_t row, size_t col, const mxArray *data)
{
  mxArray *pm = mxGetCell(data, 0);

  double *val = mxGetPr(pm);
  mwIndex *ind = mxGetIr(pm);
  mwIndex *ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);

  int nrows = dims[0];
  int ncols = dims[1];
  int nnz = ptr[ncols];

  // convert from mwIndex -> int32_t, double -> real
  std::vector<real> vec_val(val, val + nnz);
  std::vector<int32_t> vec_ptr(ptr, ptr + (ncols + 1));
  std::vector<int32_t> vec_ind(ind, ind + nnz); 

  return BlockSparse<real>::CreateFromCSC(
    row, 
    col,
    nrows,
    ncols,
    nnz,
    vec_val,
    vec_ptr,
    vec_ind);
}

BackendPDHG<real>* 
CreateBackendPDHG(const mxArray *data)
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
    throw Exception("Couldn't recognize step-size variant.");

  PDHG_stepsize_cb_handle = mxGetField(data, 0, "stepsize_callback");

  BackendPDHG<real> *backend = new BackendPDHG<real>(opts);
  backend->SetStepsizeCallback(PDHGStepsizeCallback);

  return backend;
}
    
std::shared_ptr<Prox<real> >
CreateProx(const mxArray *pm) 
{
  std::string name(mxArrayToString(mxGetCell(pm, 0)));
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  
  size_t idx = (size_t) mxGetScalar(mxGetCell(pm, 1));
  size_t size = (size_t) mxGetScalar(mxGetCell(pm, 2));
  bool diagsteps = (bool) mxGetScalar(mxGetCell(pm, 3));
  mxArray *data = mxGetCell(pm, 4);
        
  Prox<real> *prox = nullptr;

  for(size_t i = 0; prox_reg[i].create_fn != nullptr; i++)
    if(prox_reg[i].name.compare(name) == 0)
      prox = prox_reg[i].create_fn(idx, size, diagsteps, data);

  if(!prox) // prox not found -> look in custom registry
    for(size_t i = 0; custom_prox_reg[i].create_fn != nullptr; i++)
      if(custom_prox_reg[i].name.compare(name) == 0)
        prox = custom_prox_reg[i].create_fn(idx, size, diagsteps, data);

  if(!prox) // prox with that name does not exist
  {
    std::ostringstream ss;
    ss << "MexFactory::CreateProx failed. Prox with ID '" << name.c_str() << "' not registered in ProxFactory.";

    throw Exception(ss.str());
  }

  return std::shared_ptr<Prox<real> >(prox);
}

std::shared_ptr<Block<real> >
CreateBlock(const mxArray *pm)
{
  std::string name(mxArrayToString(mxGetCell(pm, 0)));
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  size_t row = (size_t) mxGetScalar(mxGetCell(pm, 1));
  size_t col = (size_t) mxGetScalar(mxGetCell(pm, 2));
  mxArray *data = mxGetCell(pm, 3);

  Block<real> *block = nullptr;

  for(size_t i = 0; block_reg[i].create_fn != nullptr; i++)
    if(block_reg[i].name.compare(name) == 0)
      block = block_reg[i].create_fn(row, col, data);

  if(!block) // block not found -> look in custom registry
    for(size_t i = 0; custom_block_reg[i].create_fn != nullptr; i++)
      if(custom_block_reg[i].name.compare(name) == 0)
        block = custom_block_reg[i].create_fn(row, col, data);

  if(!block) // block with that name does not exist
  {
    std::ostringstream ss;
    ss << "MexFactory::CreateBlock failed. Block with ID '" << name.c_str() << "' not registered in BlockFactory.";

    throw Exception(ss.str());
  }

  return std::shared_ptr<Block<real> >(block);
}
    
std::shared_ptr<Backend<real> >
CreateBackend(const mxArray *pm)
{
  std::string name(mxArrayToString(mxGetCell(pm, 0)));
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  Backend<real> *backend = nullptr;

  for(size_t i = 0; backend_reg[i].create_fn != nullptr; i++)
    if(backend_reg[i].name.compare(name) == 0)
      backend = backend_reg[i].create_fn(mxGetCell(pm, 1));

  if(!backend)
  {
    std::ostringstream ss;
    ss << "MexFactory::CreateBackend '" << name.c_str() << "' not registered in BackendFactory.";

    throw Exception(ss.str());
  }

  return std::shared_ptr<Backend<real> >(backend);
}

std::shared_ptr<Problem<real> >
CreateProblem(const mxArray *pm)
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
    throw Exception("Problem scaling variant not recognized.");

  return std::shared_ptr<Problem<real> >(prob);
}

typename Solver<real>::Options 
CreateSolverOptions(const mxArray *pm)
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

}
