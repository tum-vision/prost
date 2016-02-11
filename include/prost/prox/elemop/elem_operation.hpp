#ifndef PROST_ELEM_OPERATION_HPP_
#define PROST_ELEM_OPERATION_HPP_

#include "prost/prox/shared_mem.hpp"
#include "prost/prox/vector.hpp"

namespace prost {

// TODO: comment me
template<size_t DIM = 0, size_t COEFFS_COUNT = 0, typename SHARED_MEM_TYPE = char>
struct ElemOperation 
{
public: 
  static const size_t kCoeffsCount = COEFFS_COUNT;
  static const size_t kDim = DIM;
  struct GetSharedMemCount {
    inline __host__ __device__ size_t operator()(size_t dim) { return 0; }
  };
  typedef SHARED_MEM_TYPE SharedMemType;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_HPP_
