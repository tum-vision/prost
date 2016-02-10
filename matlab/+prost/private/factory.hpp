#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include "prost/prox/prox.hpp"
#include "prost/prox/prox_moreau.hpp"
#include "prost/prox/prox_zero.hpp"

#include "prost/linop/linearoperator.hpp"
#include "prost/linop/block.hpp"
#include "prost/linop/block_zero.hpp"
#include "prost/linop/block_sparse.hpp"

#include "prost/backend/backend.hpp"
#include "prost/backend/backend_pdhg.hpp"

#include "prost/prox/prox_elem_operation.hpp"
#include "prost/prox/elemop/elem_operation_1d.hpp"
#include "prost/prox/elemop/elem_operation_norm2.hpp"
#include "prost/prox/elemop/elem_operation_ind_simplex.hpp"
#include "prost/prox/elemop/function_1d.hpp"

#include "prost/common.hpp"
#include "prost/exception.hpp"
#include "prost/problem.hpp"
#include "prost/solver.hpp"

// has to be included at end, otherwise 
// some compiler problems with std::printf 
#include "mex.h"
#include "config.hpp"

namespace mex
{

using std::vector;
using std::string;
using std::function;
using std::shared_ptr;

void SolverIntermCallback(int iter, const vector<real>& primal, const vector<real>& dual);
     
shared_ptr<prost::Prox<real> > CreateProx(const mxArray *pm);
shared_ptr<prost::Block<real> > CreateBlock(const mxArray *pm);
shared_ptr<prost::Backend<real> > CreateBackend(const mxArray *pm);
shared_ptr<prost::Problem<real> > CreateProblem(const mxArray *pm);
typename prost::Solver<real>::Options CreateSolverOptions(const mxArray *pm);

// prox
prost::ProxMoreau<real>* CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data);   
prost::ProxZero<real>* CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data);

template<class FUN_1D> 
prost::ProxElemOperation<real, prost::ElemOperation1D<real, FUN_1D> >*
CreateProxElemOperation1D(size_t idx, size_t size, bool diagsteps, const mxArray *data);

template<class FUN_1D> 
prost::ProxElemOperation<real, prost::ElemOperationNorm2<real, FUN_1D> >*
CreateProxElemOperationNorm2(size_t idx, size_t size, bool diagsteps, const mxArray *data);

prost::ProxElemOperation<real, prost::ElemOperationIndSimplex<real> >*
CreateProxElemOperationIndSimplex(size_t idx, size_t size, bool diagsteps, const mxArray *data);

// block
prost::BlockZero<real>* CreateBlockZero(size_t row, size_t col, const mxArray *data);
prost::BlockSparse<real>* CreateBlockSparse(size_t row, size_t col, const mxArray *data);

// backend
prost::BackendPDHG<real>* CreateBackendPDHG(const mxArray *data);

struct ProxRegistry
{
  string name;
  function<prost::Prox<real>*(size_t, size_t, bool, const mxArray*)> create_fn;
};

struct BlockRegistry
{
  string name;
  function<prost::Block<real>*(size_t, size_t, const mxArray*)> create_fn;
};

struct BackendRegistry
{
  string name;
  function<prost::Backend<real>*(const mxArray*)> create_fn;
};

// user-defined custom prox operators and blocks
extern ProxRegistry custom_prox_reg[];
extern BlockRegistry custom_block_reg[];

} // namespace mex

#endif // MEX_FACTORY_HPP_
