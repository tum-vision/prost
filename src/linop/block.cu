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
void Block<T>::Initialize()
{
}

template<typename T>
void Block<T>::Release()
{
}
  
template<typename T>
void Block<T>::EvalAdd(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& rhs)
{
  EvalLocalAdd(
    result.begin() + row_,
    result.begin() + row_ + nrows_,
    rhs.cbegin() + col_,
    rhs.cbegin() + col_ + ncols_);
}

template<typename T>
void Block<T>::EvalAdjointAdd(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& rhs)
{
  EvalAdjointLocalAdd(
    result.begin() + col_,
    result.begin() + col_ + ncols_,
    rhs.cbegin() + row_,
    rhs.cbegin() + row_ + nrows_);
}

// Explicit template instantiation
template class Block<float>;
template class Block<double>;
