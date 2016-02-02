#include "mex_factory.hpp"

using namespace prox;
using namespace elemop;
using namespace std;

void MexFactory::Init() {
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DZero<real>>>>("elem_operation:1d:zero", CreateProxElemOperation1D<Function1DZero<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DAbs<real>>>>("elem_operation:1d:abs", CreateProxElemOperation1D<Function1DAbs<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DSquare<real>>>>("elem_operation:1d:square", CreateProxElemOperation1D<Function1DSquare<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DIndLeq0<real>>>>("elem_operation:1d:ind_leq0", CreateProxElemOperation1D<Function1DIndLeq0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DIndGeq0<real>>>>("elem_operation:1d:ind_geq0", CreateProxElemOperation1D<Function1DIndGeq0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DIndEq0<real>>>>("elem_operation:1d:ind_eq0", CreateProxElemOperation1D<Function1DIndEq0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DIndBox01<real>>>>("elem_operation:1d:ind_box01", CreateProxElemOperation1D<Function1DIndBox01<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DMaxPos0<real>>>>("elem_operation:1d:max_pos0", CreateProxElemOperation1D<Function1DMaxPos0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DL0<real>>>>("elem_operation:1d:l0", CreateProxElemOperation1D<Function1DL0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DHuber<real>>>>("elem_operation:1d:huber", CreateProxElemOperation1D<Function1DHuber<real>>);
                                                                                                                                    
                                                                                                                           
                                                                                                                                    
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DZero<real>>>>("elem_operation:norm2:zero", CreateProxElemOperationNorm2<Function1DZero<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DAbs<real>>>>("elem_operation:norm2:abs", CreateProxElemOperationNorm2<Function1DAbs<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DSquare<real>>>>("elem_operation:norm2:square", CreateProxElemOperationNorm2<Function1DSquare<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DIndLeq0<real>>>>("elem_operation:norm2:ind_leq0", CreateProxElemOperationNorm2<Function1DIndLeq0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DIndGeq0<real>>>>("elem_operation:norm2:ind_geq0", CreateProxElemOperationNorm2<Function1DIndGeq0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DIndEq0<real>>>>("elem_operation:norm2:ind_eq0", CreateProxElemOperationNorm2<Function1DIndEq0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DIndBox01<real>>>>("elem_operation:norm2:ind_box01", CreateProxElemOperationNorm2<Function1DIndBox01<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DMaxPos0<real>>>>("elem_operation:norm2:max_pos0", CreateProxElemOperationNorm2<Function1DMaxPos0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DL0<real>>>>("elem_operation:norm2:l0", CreateProxElemOperationNorm2<Function1DL0<real>>);
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, Function1DHuber<real>>>>("elem_operation:norm2:huber", CreateProxElemOperationNorm2<Function1DHuber<real>>);
    
    
    ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationSimplex<real>>>("elem_operation:simplex", CreateProxElemOperationSimplex);
    
    ProxFactory::GetInstance()->Register<ProxMoreau<real>>("moreau", CreateProxMoreau);
    ProxFactory::GetInstance()->Register<ProxZero<real>>("zero", CreateProxZero);
}

template<size_t COEFFS_COUNT>
void MexFactory::GetCoefficients(array<vector<real>, COEFFS_COUNT>& coeffs, const mxArray *coeffs_mx, size_t count) {
  for(size_t i = 0; i < COEFFS_COUNT; i++) {
      
    const mwSize *dims = mxGetDimensions(mxGetCell(coeffs_mx, i));
    
    if(dims[0] != 1 && dims[0] != count) {
      mexErrMsgTxt("Dimension of coefficients has to be equal to 1 or count\n");
      exit(-1);
    }
        
    coeffs[i].resize(dims[0]);
    double *val;
    val = mxGetPr(mxGetCell(coeffs_mx, i));
    for(size_t j = 0; j < dims[0]; j++)
      coeffs[i][j] = (real)val[j];
  }
}

template<class FUN_1D>
ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>* MexFactory::CreateProxElemOperation1D(int idx, int size, bool diagsteps, const mxArray *data) {
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 1));

  array<vector<real>, 7> coeffs;
  mxArray *coeffs_mx = mxGetCell(data, 2);
  GetCoefficients<7>(coeffs, coeffs_mx, count);

  return new ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>(idx, count, 1, interleaved, diagsteps, coeffs);   
}

template<class FUN_1D>
ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D>>* MexFactory::CreateProxElemOperationNorm2(int idx, int size, bool diagsteps, const mxArray *data) {
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  array<vector<real>, 7> coeffs;
  mxArray *coeffs_mx = mxGetCell(data, 3);
  GetCoefficients<7>(coeffs, coeffs_mx, count);
  
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

    mexPrintf("Attempting to create prox<'%s', idx=%d, size=%d, diagsteps=%d>...\n",
    name.c_str(), idx, size, diagsteps);
    
    Prox<real>* prox = nullptr;
    try {
      prox = ProxFactory::GetInstance()->Create(name, idx, size, diagsteps, data);
    } catch (const std::out_of_range& oor) {
      mexPrintf("Failed. Warning: Prox with ID '%s' not registered in ProxFactory\n", name.c_str());
      exit(-1);
      return nullptr;
    }
    mexPrintf("success.\n");
  
    return move(unique_ptr<Prox<real>>(prox));
}