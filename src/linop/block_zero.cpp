#include "linop/block_zero.hpp"

#include <iostream>
#include <cuda_runtime.h>

using namespace linop;

template<typename T>
BlockZero<T>::BlockZero(size_t row, size_t col, size_t nrows, size_t ncols) : Block<T>(row, col, nrows, ncols) {}      

template<typename T>
void BlockZero<T>::EvalLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end) {
  // do nothing for zero operator
}

template<typename T>
void BlockZero<T>::EvalAdjointLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end) {
  
  // do nothing for zero operator
}

template<typename T>
T BlockZero<T>::row_sum(size_t row, T alpha) const {
  return 0;
}

template<typename T>
T BlockZero<T>::col_sum(size_t col, T alpha) const {
  return 0;
}


template<typename T>
size_t BlockZero<T>::gpu_mem_amount() const {
    return 0;
}

// Explicit template instantiation
template class BlockZero<float>;
template class BlockZero<double>;

