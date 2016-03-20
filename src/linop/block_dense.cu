#include "prost/linop/block_dense.hpp"
#include "prost/exception.hpp"

namespace prost
{

template<> cublasHandle_t BlockDense<float>::cublas_handle_ = nullptr;
template<> cublasHandle_t BlockDense<double>::cublas_handle_ = nullptr;

template<typename T>
BlockDense<T>::BlockDense(size_t row, size_t col, size_t nrows, size_t ncols)
    : Block<T>(row, col, nrows, ncols)
{
}

template<typename T>
BlockDense<T>* BlockDense<T>::CreateFromColFirstData(
    size_t row, size_t col, size_t nrows, size_t ncols, const std::vector<T>& data)
{
  BlockDense<T> *block = new BlockDense<T>(row, col, nrows, ncols);
  block->host_data_ = data;

  return block;
}

template<typename T>
void BlockDense<T>::Initialize()
{
  if(cublas_handle_ == nullptr)
    cublasCreate_v2(&cublas_handle_);

  data_.resize(this->nrows() * this->ncols());
  thrust::copy(host_data_.begin(), host_data_.end(), data_.begin());
}

template<typename T>
T BlockDense<T>::row_sum(size_t row, T alpha) const
{
  T sum = 0;
  for(size_t c = 0; c < this->ncols(); ++c)
    sum += std::pow(std::abs(host_data_[c * this->nrows() + row]), alpha);
  
  return sum;
}

template<typename T>
T BlockDense<T>::col_sum(size_t col, T alpha) const
{
  T sum = 0;
  for(size_t r = 0; r < this->nrows(); ++r)
    sum += std::pow(std::abs(host_data_[col * this->nrows() + r]), alpha);

  return sum;
}

template<typename T>
size_t BlockDense<T>::gpu_mem_amount() const
{
  return this->nrows() * this->ncols() * sizeof(T);
}

template<>
void BlockDense<float>::EvalLocalAdd(
    const typename device_vector<float>::iterator& res_begin,
    const typename device_vector<float>::iterator& res_end,
    const typename device_vector<float>::const_iterator& rhs_begin,
    const typename device_vector<float>::const_iterator& rhs_end)
{
  static const float alpha = 1.f;
  static const float beta = 1.f;

  cublasStatus_t status = cublasSgemv(cublas_handle_,
                                      CUBLAS_OP_N,
                                      static_cast<int>(this->nrows()),
                                      static_cast<int>(this->ncols()),
                                      &alpha,
                                      thrust::raw_pointer_cast(data_.data()),
                                      static_cast<int>(this->nrows()),
                                      thrust::raw_pointer_cast(&(*rhs_begin)),
                                      1,
                                      &beta,
                                      thrust::raw_pointer_cast(&(*res_begin)),
                                      1);

  if(status != CUBLAS_STATUS_SUCCESS)
    throw Exception("BlockDense::EvalLocalAdd failed.");
}

template<>
void BlockDense<double>::EvalLocalAdd(
    const typename device_vector<double>::iterator& res_begin,
    const typename device_vector<double>::iterator& res_end,
    const typename device_vector<double>::const_iterator& rhs_begin,
    const typename device_vector<double>::const_iterator& rhs_end)
{
  static const double alpha = 1.f;
  static const double beta = 1.f;

  cublasStatus_t status = cublasDgemv(cublas_handle_,
                                      CUBLAS_OP_N,
                                      static_cast<int>(this->nrows()),
                                      static_cast<int>(this->ncols()),
                                      &alpha,
                                      thrust::raw_pointer_cast(data_.data()),
                                      static_cast<int>(this->nrows()),
                                      thrust::raw_pointer_cast(&(*rhs_begin)),
                                      1,
                                      &beta,
                                      thrust::raw_pointer_cast(&(*res_begin)),
                                      1);

  if(status != CUBLAS_STATUS_SUCCESS)
    throw Exception("BlockDense::EvalLocalAdd failed.");
}

template<>
void BlockDense<float>::EvalAdjointLocalAdd(
    const typename device_vector<float>::iterator& res_begin,
    const typename device_vector<float>::iterator& res_end,
    const typename device_vector<float>::const_iterator& rhs_begin,
    const typename device_vector<float>::const_iterator& rhs_end)
{
  static const float alpha = 1.f;
  static const float beta = 1.f;

  cublasStatus_t status = cublasSgemv(cublas_handle_,
                                      CUBLAS_OP_T,
                                      static_cast<int>(this->nrows()),
                                      static_cast<int>(this->ncols()),
                                      &alpha,
                                      thrust::raw_pointer_cast(data_.data()),
                                      static_cast<int>(this->nrows()),
                                      thrust::raw_pointer_cast(&(*rhs_begin)),
                                      1,
                                      &beta,
                                      thrust::raw_pointer_cast(&(*res_begin)),
                                      1);

  if(status != CUBLAS_STATUS_SUCCESS)
    throw Exception("BlockDense::EvalLocalAdd failed.");
}

template<>
void BlockDense<double>::EvalAdjointLocalAdd(
    const typename device_vector<double>::iterator& res_begin,
    const typename device_vector<double>::iterator& res_end,
    const typename device_vector<double>::const_iterator& rhs_begin,
    const typename device_vector<double>::const_iterator& rhs_end)
{
  static const double alpha = 1.f;
  static const double beta = 1.f;

  cublasStatus_t status = cublasDgemv(cublas_handle_,
                                      CUBLAS_OP_T,
                                      static_cast<int>(this->nrows()),
                                      static_cast<int>(this->ncols()),
                                      &alpha,
                                      thrust::raw_pointer_cast(data_.data()),
                                      static_cast<int>(this->nrows()),
                                      thrust::raw_pointer_cast(&(*rhs_begin)),
                                      1,
                                      &beta,
                                      thrust::raw_pointer_cast(&(*res_begin)),
                                      1);

  if(status != CUBLAS_STATUS_SUCCESS)
    throw Exception("BlockDense::EvalLocalAdd failed.");
}

// Explicit template instantiation
template class BlockDense<float>;
template class BlockDense<double>;

} // namespace prost
