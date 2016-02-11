#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include <memory>

#include "linop/linearoperator.hpp"
#include "linop/block.hpp"
#include "linop/block_zero.hpp"
#include "linop/block_sparse.hpp"

#include "backend/backend.hpp"
#include "backend/backend_pdhg.hpp"

#include "problem.hpp"
#include "solver.hpp"

#include "prox/prox.hpp"
#include "prox/prox_moreau.hpp"
#include "prox/prox_zero.hpp"
#include "prox/prox_elem_operation.hpp"
#include "prox/elemop/elem_operation_1d.hpp"
#include "prox/elemop/elem_operation_norm2.hpp"
#include "prox/elemop/elem_operation_simplex.hpp"
#include "prox/proj_epi_quadratic_fun.hpp"

// has to be included at end, otherwise 
// some compiler problems with std::printf 
#include "mex.h"
#include "mex_config.hpp"

namespace mex_factory
{

void SolverIntermCallback(int iter, const std::vector<real>& primal, const std::vector<real>& dual);
     
std::shared_ptr<Prox<real> > CreateProx(const mxArray *pm);
std::shared_ptr<Block<real> > CreateBlock(const mxArray *pm);
std::shared_ptr<Backend<real> > CreateBackend(const mxArray *pm);
std::shared_ptr<Problem<real> > CreateProblem(const mxArray *pm);
typename Solver<real>::Options CreateSolverOptions(const mxArray *pm);

// prox
ProxMoreau<real>* CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data);   
ProxZero<real>* CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data);

template<class FUN_1D> 
ProxElemOperation<real, ElemOperation1D<real, FUN_1D> >*
CreateProxElemOperation1D(size_t idx, size_t size, bool diagsteps, const mxArray *data);

template<class FUN_1D> 
ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D> >*
CreateProxElemOperationNorm2(size_t idx, size_t size, bool diagsteps, const mxArray *data);

ProxElemOperation<real, ElemOperationSimplex<real> >*
CreateProxElemOperationSimplex(size_t idx, size_t size, bool diagsteps, const mxArray *data);

ProjEpiQuadraticFun<real>*
CreateProjEpiQuadraticFun(size_t idx, size_t size, bool diagsteps, const mxArray *data);

// block
BlockZero<real>* CreateBlockZero(size_t row, size_t col, const mxArray *data);
BlockSparse<real>* CreateBlockSparse(size_t row, size_t col, const mxArray *data);

// backend
BackendPDHG<real>* CreateBackendPDHG(const mxArray *data);

struct ProxRegistry
{
  std::string name;
  std::function<Prox<real>*(size_t, size_t, bool, const mxArray*)> create_fn;
};

struct BlockRegistry
{
  std::string name;
  std::function<Block<real>*(size_t, size_t, const mxArray*)> create_fn;
};

struct BackendRegistry
{
  std::string name;
  std::function<Backend<real>*(const mxArray*)> create_fn;
};

// user-defined custom prox operators and blocks
extern ProxRegistry custom_prox_reg[];
extern BlockRegistry custom_block_reg[];

}

#endif
