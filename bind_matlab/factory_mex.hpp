#ifndef FACTORY_MEX_HPP_
#define FACTORY_MEX_HPP_

#include "mex.h"

#include "config.hpp"
#include "solver/solver.hpp"
#include "util/sparse_matrix.hpp"

// forward declarations
template<typename T> class Prox;
template<typename T> class Prox1D;
template<typename T> class ProxNorm2;
template<typename T> class ProxEpiConjQuadr;
template<typename T> class ProxMoreau;
template<typename T> class ProxSimplex;
template<typename T> class ProxZero;

// individual prox constructors
Prox1D<real>* Prox1DFromMatlab(int idx, int count, const mxArray *data);
ProxNorm2<real>* ProxNorm2FromMatlab(int idx, int count, int dim, bool interleaved, const mxArray *data);
ProxEpiConjQuadr<real>* ProxEpiConjQuadrFromMatlab(int idx, int count, bool interleaved, const mxArray *data);
ProxMoreau<real>* ProxMoreauFromMatlab(const mxArray *data);
ProxSimplex<real>* ProxSimplexFromMatlab(int idx, int count, int dim, bool interleaved, const mxArray *data);
ProxZero<real>* ProxZeroFromMatlab(int idx, int count);

// TODO: write a proper factory class for this?
Prox<real>* ProxFromMatlab(const mxArray *pm);
SparseMatrix<real>* MatrixFromMatlab(const mxArray *pm);
void SolverOptionsFromMatlab(const mxArray *pm, SolverOptions& opts, mxArray **cb_func_handle);

LinOp<real>* LinOpFromMatlab(

#endif
