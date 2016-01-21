#ifndef MEX_FACTORY_HPP_
#define MEX_FACTORY_HPP_

#include "factory.hpp"

#include "prox/prox.hpp"
#include "prox/prox_moreau.hpp"
#include "prox/prox_zero.hpp"

#include "linop/linearoperator.hpp"
#include "linop/block.hpp"
#include "linop/block_zero.hpp"

#include <algorithm>
#include <memory>
#include <string>

// has to be included at end, otherwise 
// some compiler problems with std::printf 
#include "mex.h"
#include "mex_config.hpp"

typedef Factory<Prox<real>, int, int, bool, const mxArray*> ProxFactory;
typedef Factory<Block<real>, int, int, const mxArray*> BlockFactory;

class MexFactory 
{
public:
  static void Init() 
  {
    ProxFactory::GetInstance()->Register<ProxMoreau<real> >("moreau", CreateProxMoreau);
    ProxFactory::GetInstance()->Register<ProxZero<real> >("zero", CreateProxZero);
  }
    
  static ProxMoreau<real>* CreateProxMoreau(int idx, int size, bool diagsteps, const mxArray *data) 
  {
    return new ProxMoreau<real>(std::unique_ptr<Prox<real> >(CreateProx(mxGetCell(data, 0))));
  }
    
  static ProxZero<real>* CreateProxZero(int idx, int size, bool diagsteps, const mxArray *data) 
  {
    return new ProxZero<real>(idx, size);
  }
    
  static Prox<real>* CreateProx(const mxArray *pm) 
  {
    std::string name(mxArrayToString(mxGetCell(pm, 0)));
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    size_t idx = (size_t) mxGetScalar(mxGetCell(pm, 1));
    size_t size = (size_t) mxGetScalar(mxGetCell(pm, 2));
    bool diagsteps = (bool) mxGetScalar(mxGetCell(pm, 3));
    mxArray *data = mxGetCell(pm, 4);
        
    return ProxFactory::GetInstance()->Create(name, idx, size, diagsteps, data);
  }
};
#endif
