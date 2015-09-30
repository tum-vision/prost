#include "mex_factory.hpp"

#include <algorithm>

#include "prox/prox.hpp"
#include "prox/prox_1d.hpp"
#include "prox/prox_epi_conjquadr.hpp"
#include "prox/prox_moreau.hpp"
#include "prox/prox_norm2.hpp"
#include "prox/prox_simplex.hpp"
#include "prox/prox_zero.hpp"

#include "linop/linop.hpp"
#include "linop/linop_gradient.hpp"
#include "linop/linop_identity.hpp"
#include "linop/linop_sparse.hpp"

/**
 * @brief Returns the prox-function corresponding to the string.
 */
Prox1DFunction Prox1DFunctionFromString(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  static std::string names[] = {
    "zero",
    "abs",
    "square",
    "max_pos0",
    "ind_leq0",
    "ind_geq0",
    "ind_eq0",
    "ind_box01" };

  static Prox1DFunction funcs[] = {
    kZero,
    kAbs,
    kSquare,
    kMaxPos0,
    kIndLeq0,
    kIndGeq0,
    kIndEq0,
    kIndBox01 };

  for(int i = 0; i < kNumProx1DFunctions; i++)
    if(names[i] == name)
      return funcs[i];

  return kInvalidProx;
}


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
  int32_t *ptr_int = new int32_t[n + 1];
  int32_t *ind_int = new int32_t[nnz];

  uint64_t max_ind = 0, max_ptr = 0;
  for(int i = 0; i < nnz; i++) {
    val_real[i] = (real)val[i];
    ind_int[i] = (int32_t)ind[i];
  }
  
  for(int i = 0; i < n + 1; i++) {
    ptr_int[i] = (int32_t) ptr[i];

    assert(ptr_int[i] >= 0 && ptr_int[i] <= nnz); 
  }
  
  SparseMatrix<real> *mat = SparseMatrix<real>::CreateFromCSC(
      m, n, nnz, val_real, ptr_int, ind_int);

  return mat;
}

/**
 * @brief Reads an option struct from MATLAB.
 */
// TODO: handle non-existing fields
void SolverOptionsFromMatlab(const mxArray *pm, SolverOptions& opts, mxArray **cb_func_handle) {

  std::string be_name(mxArrayToString(mxGetField(pm, 0, "backend")));
  std::string adapt_type(mxArrayToString(mxGetField(pm, 0, "adapt")));
  std::string precond(mxArrayToString(mxGetField(pm, 0, "precond")));

  std::transform(be_name.begin(),
                 be_name.end(),
                 be_name.begin(),
                 ::tolower);
  
  std::transform(adapt_type.begin(),
                 adapt_type.end(),
                 adapt_type.begin(),
                 ::tolower);

  std::transform(precond.begin(),
                 precond.end(),
                 precond.begin(),
                 ::tolower);
  
  opts.max_iters = (int) mxGetScalar(mxGetField(pm, 0, "max_iters"));
  opts.cb_iters = (int) mxGetScalar(mxGetField(pm, 0, "cb_iters"));
  opts.tol_primal = (real) mxGetScalar(mxGetField(pm, 0, "tol_primal"));
  opts.tol_dual = (real) mxGetScalar(mxGetField(pm, 0, "tol_dual"));
  opts.ad_strong.gamma = (real) mxGetScalar(mxGetField(pm, 0, "ads_gamma"));
  opts.ad_balance.alpha0 = (real) mxGetScalar(mxGetField(pm, 0, "adb_alpha0"));
  opts.ad_balance.nu = (real) mxGetScalar(mxGetField(pm, 0, "adb_nu"));
  opts.ad_balance.delta = (real) mxGetScalar(mxGetField(pm, 0, "adb_delta"));
  opts.ad_balance.s = (real) mxGetScalar(mxGetField(pm, 0, "adb_s"));
  opts.ad_converge.delta = (real) mxGetScalar(mxGetField(pm, 0, "adc_delta"));
  opts.ad_converge.tau = (real) mxGetScalar(mxGetField(pm, 0, "adc_tau"));
  opts.verbose = (bool) mxGetScalar(mxGetField(pm, 0, "verbose"));
  opts.bt_enabled = (bool) mxGetScalar(mxGetField(pm, 0, "bt_enabled"));
  opts.bt_beta = (real) mxGetScalar(mxGetField(pm, 0, "bt_beta"));
  opts.bt_gamma = (real) mxGetScalar(mxGetField(pm, 0, "bt_gamma"));
  opts.precond_alpha = (real) mxGetScalar(mxGetField(pm, 0, "precond_alpha"));

  if("pdhg" == be_name)
    opts.backend = kBackendPDHG;
  else
    mexErrMsgTxt("Unknown backend.");

  if("none" == adapt_type)
    opts.adapt = kAdaptNone;
  else if("strong" == adapt_type)
    opts.adapt = kAdaptStrong;
  else if("balance" == adapt_type)
    opts.adapt = kAdaptBalance;
  else if("converge" == adapt_type)
    opts.adapt = kAdaptConverge;
  else
    mexErrMsgTxt("Unknown PDHG variant.");

  if("off" == precond)
    opts.precond = kPrecondScalar;
  else if("alpha" == precond)
    opts.precond = kPrecondAlpha;
  else if("equil" == precond)
    opts.precond = kPrecondEquil;
  else
    mexErrMsgTxt("Unknown Preconditioner.");

  *cb_func_handle = mxGetField(pm, 0, "callback");
}

/**
 * @brief ...
 */
void Prox1DCoefficientsFromMatlab(const mxArray *pm, Prox1DCoefficients<real>& coeffs) { 
  const mwSize *dims;
  double *val;

  std::vector<real>* coeff_array[7] = {
    &coeffs.a,
    &coeffs.b,
    &coeffs.c,
    &coeffs.d,
    &coeffs.e,
    &coeffs.alpha,
    &coeffs.beta };

  // Loop starts at 1 because cell 0 is prox-name.
  for(int i = 1; i <= 7; ++i) {
    dims = mxGetDimensions(mxGetCell(pm, i));
    val = mxGetPr(mxGetCell(pm, i));

    for(int j = 0; j < dims[0]; j++)
      (*coeff_array[i - 1]).push_back((real)val[j]);
  }
}

/**
 * @brief ...
 */
Prox1D<real> *Prox1DFromMatlab(
    int idx,
    int count,
    const mxArray *data)
{
  std::string func_name(mxArrayToString(mxGetCell(data, 0)));
  Prox1DFunction func = Prox1DFunctionFromString(func_name);

  if(kInvalidProx == func)
    return 0;

  Prox1DCoefficients<real> prox_coeffs;
  Prox1DCoefficientsFromMatlab(data, prox_coeffs);

  return new Prox1D<real>(idx, count, prox_coeffs, func);
}

/**
 * @brief ...
 */
ProxNorm2<real> *ProxNorm2FromMatlab(
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

  Prox1DCoefficients<real> prox_coeffs;
  Prox1DCoefficientsFromMatlab(data, prox_coeffs);
  
  return new ProxNorm2<real>(idx, count, dim, interleaved, prox_coeffs, func);
}

/**
 * @brief ...
 */
ProxEpiConjQuadr<real>* ProxEpiConjQuadrFromMatlab(
    int idx,
    int count,
    bool interleaved,
    const mxArray *data)
{
  EpiConjQuadrCoeffs<real> coeffs;

  const mwSize *dims;
  double *val;

  std::vector<real>* coeff_array[PROX_EPI_CONJQUADR_NUM_COEFFS] = {
    &coeffs.a,
    &coeffs.b,
    &coeffs.c,
    &coeffs.alpha,
    &coeffs.beta };

  for(int i = 0; i < PROX_EPI_CONJQUADR_NUM_COEFFS; ++i) {
    dims = mxGetDimensions(mxGetCell(data, i));
    val = mxGetPr(mxGetCell(data, i));

    for(int j = 0; j < dims[0]; j++)
      (*coeff_array[i]).push_back((real)val[j]);
  }
  
  return new ProxEpiConjQuadr<real>(idx, count, interleaved, coeffs);
}

/**
 * @brief ...
 */
ProxMoreau<real>* ProxMoreauFromMatlab(const mxArray *data) {
  return new ProxMoreau<real>(ProxFromMatlab(mxGetCell(data, 0)));
}

/**
 * @brief ...
 */
ProxSimplex<real>* ProxSimplexFromMatlab(
    int idx,
    int count,
    int dim,
    bool interleaved,
    const mxArray *data)
{
  const mwSize *dims;
  double *val;
  
  dims = mxGetDimensions(mxGetCell(data, 0));
  val = mxGetPr(mxGetCell(data, 0));

  std::vector<real> coeffs;
  
  for(int j = 0; j < dims[0]; j++)
    coeffs.push_back((real)val[j]);
  
  return new ProxSimplex<real>(idx, count, dim, interleaved, coeffs);
}

/**
 * @brief ...
 */
ProxZero<real>* ProxZeroFromMatlab(int idx, int count) {
  return new ProxZero<real>(idx, count);
}


/**
 * @brief ...
 */
Prox<real>* ProxFromMatlab(const mxArray *pm) {
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
  
  Prox<real> *p = NULL;
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
  else if("zero" == name)
    p = ProxZeroFromMatlab(idx, count);

  if(NULL == p) 
    mexPrintf("Failure creating prox<'%s',idx=%d,cnt=%d,dim=%d,interleaved=%d,diagsteps=%d>...", name.c_str(), idx, count, dim, interleaved, diagsteps);

  return p;
}

LinearOperator<real>* LinearOperatorFromMatlab(const mxArray *pm) {
  const mwSize *dims = mxGetDimensions(pm);
  size_t num_linops = dims[0];

  LinearOperator<real> *result =
      new LinearOperator<real>();
  for(size_t i = 0; i < num_linops; i++) {
    mxArray *cell = mxGetCell(pm, i);
    
    std::string name(mxArrayToString(mxGetCell(cell, 0)));
    transform(name.begin(), name.end(), name.begin(), ::tolower);

    size_t row = (size_t) mxGetScalar(mxGetCell(cell, 1));
    size_t col = (size_t) mxGetScalar(mxGetCell(cell, 2));
    mxArray *data = mxGetCell(cell, 3);

    LinOp<real> *linop = NULL;
    if("gradient_2d" == name)
      linop = LinOpGradient2DFromMatlab(row, col, data);
    else if("gradient_3d" == name)
      linop = LinOpGradient3DFromMatlab(row, col, data);
    else if("sparse" == name)
      linop = LinOpSparseFromMatlab(row, col, data);
    else if("zero" == name)
      linop = LinOpZeroFromMatlab(row, col, data);
    else if("identity" == name)
      linop = LinOpIdentityFromMatlab(row, col, data);
    else if("data_prec" == name)
      linop = LinOpDataPrecFromMatlab(row, col, data);

    if(NULL == linop)
      mexErrMsgTxt("Error creating linop!");
    else
      result->AddOperator(linop);
  }

  return result;
}

LinOpIdentity<real>* LinOpIdentityFromMatlab(size_t row, size_t col, const mxArray *pm)
{  
  return NULL;
}

LinOpSparse<real>* LinOpSparseFromMatlab(size_t row, size_t col, const mxArray *pm)
{
  SparseMatrix<real> *mat = MatrixFromMatlab(mxGetCell(pm, 0));
  
  return new LinOpSparse<real>(row, col, mat);
}

LinOpGradient2D<real>* LinOpGradient2DFromMatlab(size_t row, size_t col, const mxArray *pm)
{
  size_t nx = (size_t) mxGetScalar(mxGetCell(pm, 0));
  size_t ny = (size_t) mxGetScalar(mxGetCell(pm, 1));
  size_t L = (size_t) mxGetScalar(mxGetCell(pm, 2));

  return new LinOpGradient2D<real>(row, col, nx, ny, L);
}

LinOpGradient3D<real>* LinOpGradient3DFromMatlab(size_t row, size_t col, const mxArray *pm)
{
  size_t nx = (size_t) mxGetScalar(mxGetCell(pm, 0));
  size_t ny = (size_t) mxGetScalar(mxGetCell(pm, 1));
  size_t L = (size_t) mxGetScalar(mxGetCell(pm, 2));

  return new LinOpGradient3D<real>(row, col, nx, ny, L);
}

LinOp<real>* LinOpZeroFromMatlab(size_t row, size_t col, const mxArray *pm)
{
  return NULL;
}

LinOpDataPrec<real>* LinOpDataPrecFromMatlab(size_t row, size_t col, const mxArray *pm) {
  size_t nx = (size_t) mxGetScalar(mxGetCell(pm, 0));
  size_t ny = (size_t) mxGetScalar(mxGetCell(pm, 1));
  size_t L = (size_t) mxGetScalar(mxGetCell(pm, 2));
  real left = (real) mxGetScalar(mxGetCell(pm, 3));
  real right = (real) mxGetScalar(mxGetCell(pm, 4));

  return new LinOpDataPrec<real>(row, col, nx, ny, L, left, right);    
}
