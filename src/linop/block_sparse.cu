#include <thrust/transform_reduce.h>
#include <iostream>
#include <sstream>

#include "prost/linop/block_sparse.hpp"
#include "prost/exception.hpp"

namespace prost {

template<> cusparseHandle_t BlockSparse<float>::cusp_handle_ = nullptr;
template<> cusparseHandle_t BlockSparse<double>::cusp_handle_ = nullptr;

// TODO: this is really awful C code. rewrite this.
template<typename T>
void BlockSparse<T>::csr2csc(
  int n, int m, int nz, 
  T *a, int *col_idx, int *row_start,
  T *csc_a, int *row_idx, int *col_start)
{
  int i, j, k, l;
  int *ptr;

  for (i=0; i<=m; i++) col_start[i] = 0;

  /* determine column lengths */
  for (i=0; i<nz; i++) col_start[col_idx[i]+1]++;
  for (i=0; i<m; i++) col_start[i+1] += col_start[i];

  /* go through the structure once more. Fill in output matrix. */
  for (i=0, ptr=row_start; i<n; i++, ptr++)
    for (j=*ptr; j<*(ptr+1); j++){
      k = col_idx[j];
      l = col_start[k]++;
      row_idx[l] = i;
      if (a) csc_a[l] = a[j];
    }

  /* shift back col_start */
  for (i=m; i>0; i--) col_start[i] = col_start[i-1];

  col_start[0] = 0;
}

template<typename T>
BlockSparse<T>* BlockSparse<T>::CreateFromCSC(
  size_t row,
  size_t col,
  int m,
  int n,
  int nnz,
  const std::vector<T>& val,
  const std::vector<int32_t>& ptr,
  const std::vector<int32_t>& ind)
{
  BlockSparse<T> *block = new BlockSparse<T>(row, col, m, n);
  block->nnz_ = nnz;

  // create data on host
  block->host_ind_t_ = ind; 
  block->host_ptr_t_ = ptr; 
  block->host_val_t_ = val; 

  block->host_ind_.resize(block->nnz_);
  block->host_val_.resize(block->nnz_);
  block->host_ptr_.resize(block->nrows() + 1);

  BlockSparse<T>::csr2csc(
    block->ncols(), 
    block->nrows(), 
    block->nnz_, 
    &block->host_val_t_[0],
    &block->host_ind_t_[0],
    &block->host_ptr_t_[0],
    &block->host_val_[0],
    &block->host_ind_[0],
    &block->host_ptr_[0]);

  return block;
}

template<typename T>
BlockSparse<T>::BlockSparse(size_t row, size_t col, size_t nrows, size_t ncols)
  : Block<T>(row, col, nrows, ncols)
{
}

template<typename T>
BlockSparse<T>::~BlockSparse()
{
}

template<typename T>
void BlockSparse<T>::Initialize()
{
  if(cusp_handle_ == nullptr)
    cusparseCreate(&cusp_handle_);

  cusparseCreateMatDescr(&descr_);
  cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

  // forward
  ind_.resize(nnz_);
  val_.resize(nnz_);
  ptr_.resize(this->nrows() + 1);

  // transpose
  ind_t_.resize(nnz_);
  val_t_.resize(nnz_);
  ptr_t_.resize(this->ncols() + 1);

  // copy to GPU
  thrust::copy(host_ind_t_.begin(), host_ind_t_.end(), ind_t_.begin());
  thrust::copy(host_ptr_t_.begin(), host_ptr_t_.end(), ptr_t_.begin());
  thrust::copy(host_val_t_.begin(), host_val_t_.end(), val_t_.begin());

  thrust::copy(host_ind_.begin(), host_ind_.end(), ind_.begin());
  thrust::copy(host_ptr_.begin(), host_ptr_.end(), ptr_.begin());
  thrust::copy(host_val_.begin(), host_val_.end(), val_.begin());
}

template<typename T>
T BlockSparse<T>::row_sum(size_t row, T alpha) const
{
  T sum = 0;

  for(int32_t i = host_ptr_[row]; i < host_ptr_[row + 1]; i++)
    sum += std::pow(std::abs(host_val_[i]), alpha);

  return sum;
}

template<typename T>
T BlockSparse<T>::col_sum(size_t col, T alpha) const
{
  T sum = 0;

  for(int32_t i = host_ptr_t_[col]; i < host_ptr_t_[col + 1]; i++)
    sum += std::pow(std::abs(host_val_t_[i]), alpha);

  return sum;
}

template<typename T>
size_t BlockSparse<T>::gpu_mem_amount() const
{
  size_t total_bytes = 0;

  total_bytes += 2 * nnz_ * sizeof(int32_t);
  total_bytes += (this->nrows() + this->ncols() + 2) * sizeof(int32_t);
  total_bytes += 2 * nnz_ * sizeof(T);

  return total_bytes;
}

template<>
void BlockSparse<float>::EvalLocalAdd(
  const typename thrust::device_vector<float>::iterator& res_begin,
  const typename thrust::device_vector<float>::iterator& res_end,
  const typename thrust::device_vector<float>::const_iterator& rhs_begin,
  const typename thrust::device_vector<float>::const_iterator& rhs_end)
{
  cusparseStatus_t stat;
  const float alpha = 1;
  const float beta = 1;

  stat = cusparseScsrmv(cusp_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    nrows(),
    ncols(),
    nnz_,
    &alpha,
    descr_,
    thrust::raw_pointer_cast(val_.data()),
    thrust::raw_pointer_cast(ptr_.data()),
    thrust::raw_pointer_cast(ind_.data()),
    thrust::raw_pointer_cast(&(*rhs_begin)),
    &beta,
    thrust::raw_pointer_cast(&(*res_begin)));

  if(stat != CUSPARSE_STATUS_SUCCESS)
  {
    std::ostringstream ss;
    ss << "Sparse Matrix-Vector multiplication failed. Error code = " << stat << ".";

    throw Exception(ss.str());
  }
}

template<>
void BlockSparse<float>::EvalAdjointLocalAdd(
  const typename thrust::device_vector<float>::iterator& res_begin,
  const typename thrust::device_vector<float>::iterator& res_end,
  const typename thrust::device_vector<float>::const_iterator& rhs_begin,
  const typename thrust::device_vector<float>::const_iterator& rhs_end)
{
  cusparseStatus_t stat;
  const float alpha = 1;
  const float beta = 1;

  stat = cusparseScsrmv(cusp_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    ncols(),
    nrows(),
    nnz_,
    &alpha,
    descr_,
    thrust::raw_pointer_cast(val_t_.data()),
    thrust::raw_pointer_cast(ptr_t_.data()),
    thrust::raw_pointer_cast(ind_t_.data()),
    thrust::raw_pointer_cast(&(*rhs_begin)),
    &beta,
    thrust::raw_pointer_cast(&(*res_begin)));

  if(stat != CUSPARSE_STATUS_SUCCESS)
  {
    std::ostringstream ss;
    ss << "Sparse Matrix-Vector multiplication failed. Error code = " << stat << ".";

    throw Exception(ss.str());
  }
}

template<>
void BlockSparse<double>::EvalLocalAdd(
  const typename thrust::device_vector<double>::iterator& res_begin,
  const typename thrust::device_vector<double>::iterator& res_end,
  const typename thrust::device_vector<double>::const_iterator& rhs_begin,
  const typename thrust::device_vector<double>::const_iterator& rhs_end)
{
  cusparseStatus_t stat;
  const double alpha = 1;
  const double beta = 1;

  stat = cusparseDcsrmv(cusp_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    nrows(),
    ncols(),
    nnz_,
    &alpha,
    descr_,
    thrust::raw_pointer_cast(val_.data()),
    thrust::raw_pointer_cast(ptr_.data()),
    thrust::raw_pointer_cast(ind_.data()),
    thrust::raw_pointer_cast(&(*rhs_begin)),
    &beta,
    thrust::raw_pointer_cast(&(*res_begin)));

  if(stat != CUSPARSE_STATUS_SUCCESS)
  {
    std::ostringstream ss;
    ss << "Sparse Matrix-Vector multiplication failed. Error code = " << stat << ".";

    throw Exception(ss.str());
  }
}

template<>
void BlockSparse<double>::EvalAdjointLocalAdd(
  const typename thrust::device_vector<double>::iterator& res_begin,
  const typename thrust::device_vector<double>::iterator& res_end,
  const typename thrust::device_vector<double>::const_iterator& rhs_begin,
  const typename thrust::device_vector<double>::const_iterator& rhs_end)
{
  cusparseStatus_t stat;
  const double alpha = 1;
  const double beta = 1;

  stat = cusparseDcsrmv(cusp_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    ncols(),
    nrows(),
    nnz_,
    &alpha,
    descr_,
    thrust::raw_pointer_cast(val_t_.data()),
    thrust::raw_pointer_cast(ptr_t_.data()),
    thrust::raw_pointer_cast(ind_t_.data()),
    thrust::raw_pointer_cast(&(*rhs_begin)),
    &beta,
    thrust::raw_pointer_cast(&(*res_begin)));

  if(stat != CUSPARSE_STATUS_SUCCESS)
  {
    std::ostringstream ss;
    ss << "Sparse Matrix-Vector multiplication failed. Error code = " << stat << ".";

    throw Exception(ss.str());
  }
}

// Explicit template instantiation
template class BlockSparse<float>;
template class BlockSparse<double>;

} // namespace prost