#ifndef ELEM_OPERATION_HPP_
#define ELEM_OPERATION_HPP_

#include "vector.hpp"
#include "shared_mem.hpp"

namespace prox {
namespace elemop {
template<size_t DIM, size_t COEFFS_COUNT>
struct ElemOperation {
public: 
  static const size_t coeffs_count = COEFFS_COUNT;
  static const size_t dim = DIM;
  static size_t shared_mem_count(size_t dim) { return 0; }
  typedef char shared_mem_type;
};
}
}


#endif
