#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include <memory>

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

namespace MexFactory
{

void Initialize();
     
shared_ptr<Prox<real> > CreateProx(const mxArray *pm);
shared_ptr<Block<real> > CreateBlock(const mxArray *pm);
shared_ptr<Backend<real> > CreateBackend(const mxArray *pm);
shared_ptr<Problem<real> > CreateProblem(const mxArray *pm);
typename Solver<real>::Options CreateSolverOptions(const mxArray *pm);

ProxMoreau<real>* CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data);   
ProxZero<real>* CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data);

BlockZero<real>* CreateBlockZero(size_t row, size_t col, const mxArray *data);
BlockSparse<real>* CreateBlockSparse(size_t row, size_t col, const mxArray *data);

BackendPDHG<real>* CreateBackendPDHG(const mxArray *data);

}

#endif
