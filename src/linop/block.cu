#include "linop/block.hpp"

template<typename T>
Block<T>::Block(size_t row, size_t col, size_t nrows, size_t ncols) 
  : row_(row), col_(col), nrows_(nrows), ncols_(ncols) 
{
}

template<typename T>
Block<T>::~Block() 
{ 
}
  
template<typename T>
void Block<T>::EvalAdd(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T> rhs)
{
//  EvalLocalAdd(&result[row_], &rhs[col_]);
}

template<typename T>
void Block<T>::EvalAdjointAdd(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T> rhs)
{
//  EvalAdjointLocalAdd(&result[col_], &rhs[row_]);
}

// Explicit template instantiation
template class Block<float>;
template class Block<double>;

