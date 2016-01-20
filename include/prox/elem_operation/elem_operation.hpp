#ifndef ELEM_OPERATION_HPP_
#define ELEM_OPERATION_HPP_

#include "vector.hpp"
#include "shared_mem.hpp"

namespace prox {
namespace elemOperation {
template<size_t DIM>
struct ElemOperation {
   static const size_t dim = DIM;
   static const size_t shared_mem_count = 0;
   typedef char shared_mem_type;

};
}
}

#endif