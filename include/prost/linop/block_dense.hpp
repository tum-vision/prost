#ifndef PROST_BLOCK_DENSE_HPP_
#define PROST_BLOCK_DENSE_HPP_

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "prost/linop/block.hpp"

namespace prost {

/// 
/// \brief Linear operator based on dense matrix.
/// 
template<typename T>
class BlockDense : public Block<T> {
  BlockDense(size_t row, size_t col, size_t nrows, size_t ncols);

public: 
  static BlockDense<T> *CreateFromColFirstData(
      size_t row, size_t col, size_t nrows, size_t ncols, const std::vector<T>& data);

  virtual ~BlockDense() { }

  virtual void Initialize();

  /// \brief Required for preconditioners, row and col are "local" 
  ///        for the operator, which means they start at 0.
  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t gpu_mem_amount() const;
  
protected:
  // TODO: implement sparse matrix multiplication on CPU
  virtual void EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

  virtual void EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

 private:
  device_vector<T> data_;
  vector<T> host_data_;
  static cublasHandle_t cublas_handle_;
};

} // namespace prost

#endif // PROST_BLOCK_DENSE_HPP_
