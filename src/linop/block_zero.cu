#include "linop/block_zero.hpp"

template<typename T>
BlockZero<T>::BlockZero(size_t row, size_t col, size_t nrows, size_t ncols) 
  : Block<T>(row, col, nrows, ncols)
{
}

template<typename T>
BlockZero<T>::~BlockZero()
{
}

template<typename T>
void BlockZero<T>::EvalLocalAdd(
  const typename thrust::device_vector<T>::iterator& res_begin,
  const typename thrust::device_vector<T>::iterator& res_end,
  const typename thrust::device_vector<T>::const_iterator& rhs_begin,
  const typename thrust::device_vector<T>::const_iterator& rhs_end)
{
  // do nothing for zero operator
}

template<typename T>
void BlockZero<T>::EvalAdjointLocalAdd(
  const typename thrust::device_vector<T>::iterator& res_begin,
  const typename thrust::device_vector<T>::iterator& res_end,
  const typename thrust::device_vector<T>::const_iterator& rhs_begin,
  const typename thrust::device_vector<T>::const_iterator& rhs_end)
{
  // do nothing for zero operator
}

// Explicit template instantiation
template class BlockZero<float>;
template class BlockZero<double>;
