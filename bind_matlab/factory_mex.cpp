#include "factory_mex.hpp"

#include "prox.hpp"
#include "prox_1d.hpp"
#include "prox_epi_conjquadr.hpp"
#include "prox_moreau.hpp"
#include "prox_norm2.hpp"
#include "prox_simplex.hpp"

/**
 * @brief Creates a SparseMatrix<real> instance from a MATLAB sparse matrix
 */
SparseMatrix<real> *MatrixFromMatlab(const mxArray *pm) {
  
  double *val = mxGetPr(pm);
  mwIndex *ind = mxGetIr(pm);
  mwIndex *ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);

  int m = dims[0];
  int n = dims[1];
  int nnz = ptr[n];

  // convert from mwIndex -> int, double -> real
  real *val_real = new real[nnz];
  int *ptr_int = new int[n + 1];
  int *ind_int = new int[nnz];

  for(int i = 0; i < nnz; i++) {
    val_real[i] = (real)val[i];
    ind_int[i] = (int)ind[i];
  }
  
  for(int i = 0; i < n + 1; i++) 
    ptr_int[i] = (int) ptr[i];
  
  SparseMatrix<real> *mat = SparseMatrix<real>::CreateFromCSC(
      m, n, nnz, val_real, ptr_int, ind_int);

  return mat;
}

/**
 * @brief Reads an option struct from MATLAB.
 */
// TODO: handle non-existing fields
void SolverOptionsFromMatlab(const mxArray *pm, SolverOptions& opts) {

  std::string be_name(mxArrayToString(mxGetField(pm, 0, "backend")));
  std::string pdhg_type(mxArrayToString(mxGetField(pm, 0, "pdhg_type")));

  std::transform(be_name.begin(),
                 be_name.end(),
                 be_name.begin(),
                 ::tolower);
  
  std::transform(pdhg_type.begin(),
                 pdhg_type.end(),
                 pdhg_type.begin(),
                 ::tolower);
  
  opts.max_iters = (int) mxGetScalar(mxGetField(pm, 0, "max_iters"));
  opts.cb_iters = (int) mxGetScalar(mxGetField(pm, 0, "cb_iters"));
  opts.tolerance = (real) mxGetScalar(mxGetField(pm, 0, "tolerance"));
  opts.gamma = (real) mxGetScalar(mxGetField(pm, 0, "gamma"));
  opts.alpha0 = (real) mxGetScalar(mxGetField(pm, 0, "alpha0"));
  opts.nu = (real) mxGetScalar(mxGetField(pm, 0, "nu"));
  opts.delta = (real) mxGetScalar(mxGetField(pm, 0, "delta"));
  opts.s = (real) mxGetScalar(mxGetField(pm, 0, "s"));
  opts.verbose = (bool) mxGetScalar(mxGetField(pm, 0, "verbose"));

  if("pdhg" == be_name)
    opts.backend = kBackendPDHG;
  else
    mexErrMsgIdAndTxt("pdsolver", "Unknown backend.");

  if("alg1" == pdhg_type)
    opts.pdhg = kPDHGAlg1;
  else if("alg2" == pdhg_type)
    opts.pdhg = kPDHGAlg2;
  else if("adapt" == pdhg_type)
    opts.pdhg = kPDHGAdapt;
  else
    mexErrMsgIdAndTxt("pdsolver", "Unknown PDHG variant.");
}

/**
 * @brief ...
 */
void Prox1DCoefficientsFromMatlab(const mxArray *pm, Prox1DCoefficients& coeffs) { 
  const mwSize *dims;
  double *val;

  std::vector<real>* coeff_array[5] = {
    &coeffs.a,
    &coeffs.b,
    &coeffs.c,
    &coeffs.d,
    &coeffs.e };

  // Loop starts at 1 because cell 0 is prox-name.
  for(int i = 1; i <= 5; ++i) {
    dims = mxGetDimensions(mxGetCell(pm, i));
    val = mxGetPr(mxGetCell(pm, i));

    for(int j = 0; j < dims[0]; j++)
      (*coeff_array[i - 1]).push_back((real)val[j]);
  }
}

/**
 * @brief ...
 */
Prox1D *Prox1DFromMatlab(
    int idx,
    int count,
    const mxArray *data)
{
  std::string func_name(mxArrayToString(mxGetCell(data, 0)));
  Prox1DFunction func = Prox1DFunctionFromString(func_name);

  if(kInvalidProx == func)
    return 0;

  Prox1DCoefficients prox_coeffs;
  Prox1DCoefficientsFromMatlab(data, prox_coeffs);

  return new Prox1D(idx, count, prox_coeffs, func);
}

/**
 * @brief ...
 */
ProxNorm2 *ProxNorm2FromMatlab(
    int idx,
    int count,
    int dim,
    bool interleaved,
    const mxArray *data)
{
  std::string func_name(mxArrayToString(mxGetCell(data, 0)));
  Prox1DFunction func = Prox1DFunctionFromString(func_name);

  if(kInvalidProx == func)
    return NULL;

  Prox1DCoefficients prox_coeffs;
  Prox1DCoefficientsFromMatlab(data, prox_coeffs);
  
  return new ProxNorm2(idx, count, dim, interleaved, prox_coeffs, func);
}

/**
 * @brief ...
 */
ProxEpiConjQuadr* ProxEpiConjQuadrFromMatlab(
    int idx,
    int count,
    bool interleaved,
    const mxArray *data)
{
  // TODO: implement me!
  
  return NULL;
}

/**
 * @brief ...
 */
ProxMoreau* ProxMoreauFromMatlab(const mxArray *data) {
  // TODO: implement me!

  return NULL;
}

/**
 * @brief ...
 */
ProxSimplex* ProxSimplexFromMatlab(
    int idx,
    int count,
    int dim,
    bool interleaved,
    const mxArray *data)
{
  // TODO: implement me!
  
  return NULL;
}

/**
 * @brief ...
 */
Prox* ProxFromMatlab(const mxArray *pm) {
  std::string name(mxArrayToString(mxGetCell(pm, 0)));
  transform(name.begin(), name.end(), name.begin(), ::tolower);

  int idx = (int) mxGetScalar(mxGetCell(pm, 1));
  int count = (int) mxGetScalar(mxGetCell(pm, 2));
  int dim = (int) mxGetScalar(mxGetCell(pm, 3));
  bool interleaved = (bool) mxGetScalar(mxGetCell(pm, 4));
  bool diagsteps = (bool) mxGetScalar(mxGetCell(pm, 5));
  mxArray *data = mxGetCell(pm, 6);

  mexPrintf("Attempting to create prox<'%s',idx=%d,cnt=%d,dim=%d,interleaved=%d,diagsteps=%d>...",
            name.c_str(), idx, count, dim, interleaved, diagsteps);
    
  Prox *p = NULL;
  if("1d" == name)
    p = Prox1DFromMatlab(idx, count, data);
  else if("norm2" == name)
    p = ProxNorm2FromMatlab(idx, count, dim, interleaved, data);
  else if("epi_conjquadr" == name)
    p = ProxEpiConjQuadrFromMatlab(idx, count, interleaved, data);
  else if("moreau" == name)
    p = ProxMoreauFromMatlab(data);
  else if("simplex" == name)
    p = ProxSimplexFromMatlab(idx, count, dim, interleaved, data);

  if(NULL == p) 
    mexPrintf(" failure!\n");
  else
    mexPrintf(" done!\n");

  return p;
}

