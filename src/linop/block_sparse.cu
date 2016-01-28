#include "block_sparse.hpp"

cusparseHandle_t BlockSparse<float>::cusp_handle_ = nullptr;
cusparseHandle_t BlockSparse<double>::cusp_handle_ = nullptr;

template<typename T>
BlockSparse<T>* BlockSparse<T>::CreateFromCSC(
  size_t row,
  size_t col,
  int m,
  int n,
  int nnz,
  T *val,
  int32_t *ptr,
  int32_t *val)
{
  BlockSparse<T> *block = new BlockSparse<T>(row, col, m, n);

  if(cusp_handle_ == nullptr)
    cusparseCreate(&cusp_handle_);

  cusparseCreateMatDescr(block->descr_);
  cusparseSetMatType(block->descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(block->descr_, CUSPARSE_INDEX_BASE_ZERO);

  block->nnz_ = nnz;
  block->ind_.reserve(2 * block->nnz_);
  block->val_.reserve(2 * block->nnz_);
  block->ptr_.reserve((block->nrows() + block->ncols()) * 2);

}

template<typename T>
BlockSparse<T>::BlockSparse(size_t row, size_t col, size_t nrows, size_t ncols)
{
}

template<typename T>
BlockSparse<T>::~BlockSparse()
{
}

template<typename T>
void BlockSparse<T>::Initialize()
{
}

template<typename T>
void BlockSparse<T>::Release()
{
}
  
template<typename T>
T BlockSparse<T>::row_sum(size_t row, T alpha) const
{
}

template<typename T>
T BlockSparse<T>::col_sum(size_t col, T alpha) const
{
}

template<typename T>
size_t BlockSparse<T>::gpu_mem_amount() const
{
}

void BlockSparse<float>::EvalLocalAdd(
  const typename thrust::device_vector<float>::iterator& res_begin,
  const typename thrust::device_vector<float>::iterator& res_end,
  const typename thrust::device_vector<float>::const_iterator& rhs_begin,
  const typename thrust::device_vector<float>::const_iterator& rhs_end)
{
}

void BlockSparse<float>::EvalAdjointLocalAdd(
  const typename thrust::device_vector<float>::iterator& res_begin,
  const typename thrust::device_vector<float>::iterator& res_end,
  const typename thrust::device_vector<float>::const_iterator& rhs_begin,
  const typename thrust::device_vector<float>::const_iterator& rhs_end)
{
}

void BlockSparse<double>::EvalLocalAdd(
  const typename thrust::device_vector<float>::iterator& res_begin,
  const typename thrust::device_vector<float>::iterator& res_end,
  const typename thrust::device_vector<float>::const_iterator& rhs_begin,
  const typename thrust::device_vector<float>::const_iterator& rhs_end)
{
}

void BlockSparse<double>::EvalAdjointLocalAdd(
  const typename thrust::device_vector<float>::iterator& res_begin,
  const typename thrust::device_vector<float>::iterator& res_end,
  const typename thrust::device_vector<float>::const_iterator& rhs_begin,
  const typename thrust::device_vector<float>::const_iterator& rhs_end)
{
}

// Explicit template instantiation
template class BlockSparse<float>;
template class BlockSparse<double>;
