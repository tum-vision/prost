#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_


#include <vector>
#include <algorithm>
#include <memory>


#include "../../include/config.hpp"
#include "../../include/factory.hpp"
#include "../../include/prox/prox.hpp"
#include "../../include/prox/prox_elem_operation.hpp"
#include "../../include/prox/prox_moreau.hpp"
#include "../../include/prox/prox_zero.hpp"
#include "../../include/prox/elemop/elem_operation_1d.hpp"
#include "../../include/prox/elemop/elem_operation_norm2.hpp"
#include "../../include/prox/elemop/elem_operation_simplex.hpp"
#include "../../include/prox/elemop/function_1d.hpp"


#include "mex.h"

typedef Factory<prox::Prox<real>, int, int, bool, const mxArray*> ProxFactory;
// typedef Factory<LinOperator, int, int, const mxArray> LinOperatorFactory;

class MexFactory {
public:
    static void Init();
    
    template<class COEFFS_1D>
    static void GetCoefficients1D(vector<COEFFS_1D>& coeffs, const mxArray *coeffs_mx);
    
    template<class FUN_1D>
    static prox::ProxElemOperation<real, prox::elemop::ElemOperation1D<real, FUN_1D>>* CreateProxElemOperation1D(int idx, int size, bool diagsteps, const mxArray *data);
  
    template<class FUN_1D>
    static prox::ProxElemOperation<real, prox::elemop::ElemOperationNorm2<real, FUN_1D>>* CreateProxElemOperationNorm2(int idx, int size, bool diagsteps, const mxArray *data);
    
    static prox::ProxElemOperation<real, prox::elemop::ElemOperationSimplex<real>>* CreateProxElemOperationSimplex(int idx, int size, bool diagsteps, const mxArray *data);
    

    static prox::ProxMoreau<real>* CreateProxMoreau(int idx, int size, bool diagsteps, const mxArray *data);
    
    static prox::ProxZero<real>* CreateProxZero(int idx, int size, bool diagsteps, const mxArray *data);
    
    static std::unique_ptr<prox::Prox<real>> CreateProx(const mxArray *pm);
};
#endif
