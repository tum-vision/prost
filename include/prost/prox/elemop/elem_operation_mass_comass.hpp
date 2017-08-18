#ifndef PROST_ELEM_OPERATION_IND_PSD_CONE_3X3_HPP_
#define PROST_ELEM_OPERATION_IND_PSD_CONE_3X3_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

///
/// \brief Provides proximal operator of the mass norm (for 2-vectors in R^4)
///
template<typename T>
struct ElemOperationMass2_4 : public ElemOperation<6, 0>
{
  __host__ __device__
  ElemOperationMass2_4(size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) { }

  inline __host__ __device__
  void operator()(
    Vector<T>& res,
    const Vector<const T>& arg,
    const Vector<const T>& tau_diag,
    T tau_scal,
    bool invert_tau)
  {
    
  }
  
};

///
/// \brief Provides proximal operator of the mass norm (for 2-vectors in R^5)
///
template<typename T>
struct ElemOperationMass2_5 : public ElemOperation<10, 0>
{
  __host__ __device__
  ElemOperationMass2_4(size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) { }
};

// TODO: co-mass norm
  
}
