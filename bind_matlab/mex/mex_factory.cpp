#include "mex_factory.hpp"

using namespace prox;
using namespace elemop;
using namespace std;

void MexFactory::Init() {
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DZero<real>>>>("elem_operation:1D:zero", CreateProxElemOperation1D<Function1DZero<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DAbs<real>>>>("elem_operation:1D:abs", CreateProxElemOperation1D<Function1DAbs<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationSimplex<real>>>("elem_operation:simplex", CreateProxElemOperationSimplex);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DZero<real>>>>("elem_operation:norm2:zero", CreateProxElemOperationNorm2<Function1DZero<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DAbs<real>>>>("elem_operation:norm2:abs", CreateProxElemOperationNorm2<Function1DAbs<real>>);
    ProxFactory::GetInstance()->Register<ProxMoreau<real>>("moreau", CreateProxMoreau);
    ProxFactory::GetInstance()->Register<ProxZero<real>>("zero", CreateProxZero);
}

template<class COEFFS_1D>
void MexFactory::GetCoefficients1D(vector<COEFFS_1D>& coeffs, const mxArray *coeffs_mx) {
  double *val;

  const mwSize *dims = mxGetDimensions(mxGetCell(coeffs_mx, 0));
  coeffs.resize(dims[0]);

  // coeffs.a
  val = mxGetPr(mxGetCell(coeffs_mx, 0));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].a = (real)val[i];


  // coeffs.b
  val = mxGetPr(mxGetCell(coeffs_mx, 1));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].b = (real)val[i];


  // coeffs.c
  val = mxGetPr(mxGetCell(coeffs_mx, 2));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].c = (real)val[i];


  // coeffs.d
  val = mxGetPr(mxGetCell(coeffs_mx, 3));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].d = (real)val[i];


  // coeffs.e
  val = mxGetPr(mxGetCell(coeffs_mx, 4));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].e = (real)val[i];


  // coeffs.alpha
  val = mxGetPr(mxGetCell(coeffs_mx, 5));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].alpha = (real)val[i];


  // coeffs.beta
  val = mxGetPr(mxGetCell(coeffs_mx, 6));
  for(size_t i = 0; i < coeffs.size(); i++)
      coeffs[i].beta = (real)val[i];
}

template<class FUN_1D>
ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>* MexFactory::CreateProxElemOperation1D(int idx, int size, bool diagsteps, const mxArray *data) {
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  vector<typename ElemOperation1D<real, FUN_1D>::Coefficients> coeffs;
  mxArray *coeffs_mx = mxGetCell(data, 3);
  GetCoefficients1D<typename ElemOperation1D<real, FUN_1D>::Coefficients>(coeffs, coeffs_mx);

  return new ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>(idx, count, dim, interleaved, diagsteps, coeffs);   
}

template<class FUN_1D>
ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D>>* MexFactory::CreateProxElemOperationNorm2(int idx, int size, bool diagsteps, const mxArray *data) {
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  vector<typename ElemOperationNorm2<real, FUN_1D>::Coefficients> coeffs;

  mxArray *coeffs_mx = mxGetCell(data, 3);
  GetCoefficients1D<typename ElemOperationNorm2<real, FUN_1D>::Coefficients>(coeffs, coeffs_mx);

  return new ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D>>(idx, count, dim, interleaved, diagsteps, coeffs);   
}

ProxElemOperation<real, ElemOperationSimplex<real>>* MexFactory::CreateProxElemOperationSimplex(int idx, int size, bool diagsteps, const mxArray *data) {
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));


  return new ProxElemOperation<real, ElemOperationSimplex<real>>(idx, count, dim, interleaved, diagsteps);   
}

ProxMoreau<real>* MexFactory::CreateProxMoreau(int idx, int size, bool diagsteps, const mxArray *data) {
    return new ProxMoreau<real>(move(CreateProx(mxGetCell(data, 0))));
}

ProxZero<real>* MexFactory::CreateProxZero(int idx, int size, bool diagsteps, const mxArray *data) {
    return new ProxZero<real>(idx, size);
}

unique_ptr<Prox<real>> MexFactory::CreateProx(const mxArray *pm) {
    std::string name(mxArrayToString(mxGetCell(pm, 0)));
    transform(name.begin(), name.end(), name.begin(), ::tolower);

    int idx = (int) mxGetScalar(mxGetCell(pm, 1));
    int size = (int) mxGetScalar(mxGetCell(pm, 2));
    bool diagsteps = (bool) mxGetScalar(mxGetCell(pm, 3));
    mxArray *data = mxGetCell(pm, 4);

    mexPrintf("Attempting to create prox<'%s', idx=%d, size=%d, diagsteps=%d>...",
    name.c_str(), idx, size, diagsteps);

    return move(unique_ptr<Prox<real>>(ProxFactory::GetInstance()->Create(name, idx, size, diagsteps, data)));
}