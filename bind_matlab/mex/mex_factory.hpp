#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include "mex.h"

#include "config.hpp"
#include "solver/solver.hpp"
#include "util/sparse_matrix.hpp"

typedef Factory<Prox, int, int, bool, const mxArray*> ProxFactory;
typedef Factory<LinOperator, int, int, const mxArray> LinOperatorFactory;


class MexFactory {
public:
    static void Init() {
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DZero>>>("elem_operation:1D:zero", CreateProxElemOperation1D<Function1DZero>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperation1D<real, Function1DAbs>>>("elem_operation:1D:abs", CreateProxElemOperation1D<Function1DAbs>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationSimplex<real, 2>>>("elem_operation:simplex:2", CreateProxElemOperationSimplex<2>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationSimplex<real, 3>>>("elem_operation:simplex:3", CreateProxElemOperationSimplex<3>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, 2, Function1DZero>>>("elem_operation:norm2:zero:2", CreateProxElemOperationNorm2<2, Function1DZero>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, 3, Function1DZero>>>("elem_operation:norm2:zero:3", CreateProxElemOperationNorm2<3, Function1DZero>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, 2, Function1DAbs>>>("elem_operation:norm2:abs:2", CreateProxElemOperationNorm2<2, Function1DAbs>);
        ProxFactory::GetInstance()->Register<ProxElemOperation<real, ElemOperationNorm2<real, 3, Function1DAbs>>>("elem_operation:norm2:abs:3", CreateProxElemOperationNorm2<3, Function1DAbs>);
        ProxFactory::GetInstance()->Register<ProxMoreau<real>("moreau", CreateProxMoreau);
        ProxFactory::GetInstance()->Register<ProxZero<real>("zero", CreateProxZero);
    }
    
    static void GetCoefficients1D(vector<Coefficients1D>& coeffs, const mxArray *coeffs_mx) {
      double *val;

      const mwSize *dims = mxGetDimensions(mxGetCell(coeffs_mx, 0));
      coeffs.resize(dims[0]);
      
      // coeffs.a
      val = mxGetPr(mxGetCell(coeffs_mx, 0));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].a = (real)val[j]);
      
      
      // coeffs.b
      val = mxGetPr(mxGetCell(coeffs_mx, 1));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].b = (real)val[j]);
      
      
      // coeffs.c
      val = mxGetPr(mxGetCell(coeffs_mx, 2));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].c = (real)val[j]);
      
      
      // coeffs.d
      val = mxGetPr(mxGetCell(coeffs_mx, 3));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].d = (real)val[j]);
      
      
      // coeffs.e
      val = mxGetPr(mxGetCell(coeffs_mx, 4));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].e = (real)val[j]);
      
      
      // coeffs.alpha
      val = mxGetPr(mxGetCell(coeffs_mx, 5));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].alpha = (real)val[j]);
      
      
      // coeffs.beta
      val = mxGetPr(mxGetCell(coeffs_mx, 6));
      for(size_t i = 0; i < coeffs.size(); i++)
          coeffs[i].beta = (real)val[j]);
    }
    
    template<class FUN_1D>
    static ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>* CreateProxElemOperation1D(int idx, int size, bool diagsteps, const mxArray *data) {
      int count = (int) mxGetScalar(mxGetCell(data, 0));
      bool interleaved = (bool) mxGetScalar(mxGetCell(data, 1));
      
      vector<ElemOperation1D<real, FUN_1D>::Coefficients> coeffs;
      mxArray *coeffs_mx = mxGetCell(data, 2);
      GetCoefficients1D(coeffs, coeffs_mx);
      
      return new ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>(idx, count, interleaved, diagsteps, coeffs);   
    }
  
    template<size_t DIM>
    static ProxElemOperation<real, ElemOperationSimplex<real, DIM>>* CreateProxElemOperationSimplex(int idx, int size, bool diagsteps, const mxArray *data) {
      int count = (int) mxGetScalar(mxGetCell(data, 0));
      bool interleaved = (bool) mxGetScalar(mxGetCell(data, 1));
      
      vector<ElemOperationSimplex<real, DIM>::Coefficients> coeffs;
      
      double *val;

      const mwSize *dims = mxGetDimensions(mxGetCell(pm, 2));
      coeffs.resize(dims[0] / DIM);
      
      // coeffs.a
      val = mxGetPr(mxGetCell(pm, 2));
      for(size_t i = 0; i < dims[0] / DIM; i++)
          for(size_t j = 0; j < DIM; j++)
            coeffs[i][j] = (real)val[i * DIM + j];
    
      
      
      return new ProxElemOperation<real, ElemOperationSimplex<real, DIM>>(idx, count, interleaved, diagsteps, coeffs);   
    }
    
    template<size_t DIM, class FUN_1D>
    static ProxElemOperation<real, ElemOperationNorm2<real, DIM, FUN_1D>>* CreateProxElemOperationNorm2(int idx, int size, bool diagsteps, const mxArray *data) {
      int count = (int) mxGetScalar(mxGetCell(data, 0));
      bool interleaved = (bool) mxGetScalar(mxGetCell(data, 1));
      
      vector<ElemOperationNorm2<real, DIM, FUN_1D>::Coefficients> coeffs;
      
      mxArray *coeffs_mx = mxGetCell(data, 2);
      GetCoefficients1D(coeffs, coeffs_mx);

      return new ProxElemOperation<real, ElemOperationNorm2<real, DIM, FUN_1D>>(idx, count, interleaved, diagsteps, coeffs);   
    }
    
    static ProxMoreau<real>* CreateProxMoreau(int idx, int size, bool diagsteps, const mxArray *data) {
        return new ProxMoreau<real>(CreateProx(mxGetCell(data, 0)));
    }
    
    static ProxZero<real>* CreateProxZero(int idx, int size, bool diagsteps, const mxArray *data) {
        return new ProxZero<real>(idx, size);
    }
    
    static shared_ptr<Prox<real>> CreateProx(const mxArray *pm) {
        std::string name(mxArrayToString(mxGetCell(pm, 0)));
        transform(name.begin(), name.end(), name.begin(), ::tolower);

        int idx = (int) mxGetScalar(mxGetCell(pm, 1));
        int size = (int) mxGetScalar(mxGetCell(pm, 2));
        bool diagsteps = (bool) mxGetScalar(mxGetCell(pm, 3));
        mxArray *data = mxGetCell(pm, 4);

        mexPrintf("Attempting to create prox<'%s', idx=%d, size=%d, diagsteps=%d>...",
        name.c_str(), idx, size, diagsteps);
        
        return shared_ptr<Prox<real>>(ProxFactory::GetInstance()->Create(name, idx, size, diagsteps, data));
    }
};
#endif
