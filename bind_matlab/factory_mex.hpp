#ifndef FACTORY_MEX_HPP_
#define FACTORY_MEX_HPP_

#include "mex.h"

#include "config.hpp"
#include "solver.hpp"
#include "util/sparse_matrix.hpp"

// forward declarations - lower compilation time
class Prox;
class Prox1D;
class ProxNorm2;
class ProxEpiConjQuadr;
class ProxMoreau;
class ProxSimplex;

// individual prox constructors
Prox1D*           Prox1DFromMatlab(int idx, int count, const mxArray *data);
ProxNorm2*        ProxNorm2FromMatlab(int idx, int count, int dim, bool interleaved, const mxArray *data);
ProxEpiConjQuadr* ProxEpiConjQuadrFromMatlab(int idx, int count, bool interleaved, const mxArray *data);
ProxMoreau*       ProxMoreauFromMatlab(const mxArray *data);
ProxSimplex*      ProxSimplexFromMatlab(int idx, int count, int dim, bool interleaved, const mxArray *data);

// TODO: write a proper factory class for this
Prox*               ProxFromMatlab(const mxArray *pm);
std::vector<Prox *> ProxListFromMatlab(const mxArray *pm);

// misc things
SparseMatrix<real> *MatrixFromMatlab(const mxArray *pm);
void SolverOptionsFromMatlab(const mxArray *pm, SolverOptions& opts);

#endif
