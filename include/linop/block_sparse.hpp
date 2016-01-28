#ifndef BLOCK_SPARSE_HPP_
#define BLOCK_SPARSE_HPP_

#include <cstdlib>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

/// 
/// \brief Linear operator based on sparse matrix.
/// 
template<typename T>
class BlockSparse : public Block<T>
{
  BlockSparse(size_t row, size_t col, size_t nrows, size_t ncols);

public:
  static BlockSparse<T> *CreateFromCSC(
    size_t row,
    size_t col,
    int m,
    int n,
    int nnz,
    T *val,
    int32_t *ptr,
    int32_t *val);

  virtual ~BlockSparse();

  virtual void Initialize();
  virtual void Release();
  
  /// \brief Required for preconditioners, row and col are "local" 
  ///        for the operator, which means they start at 0.
  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t gpu_mem_amount() const;
  
protected:
  virtual void EvalLocalAdd(
    const typename thrust::device_vector<T>::iterator& res_begin,
    const typename thrust::device_vector<T>::iterator& res_end,
    const typename thrust::device_vector<T>::const_iterator& rhs_begin,
    const typename thrust::device_vector<T>::const_iterator& rhs_end);

  virtual void EvalAdjointLocalAdd(
    const typename thrust::device_vector<T>::iterator& res_begin,
    const typename thrust::device_vector<T>::iterator& res_end,
    const typename thrust::device_vector<T>::const_iterator& rhs_begin,
    const typename thrust::device_vector<T>::const_iterator& rhs_end);

  /// \brief Number of non-zero elements.
  size_t nnz_;

  static cusparseHandle_t cusp_handle_;
  cusparseMatDescr_t descr_;

  thrust::device_vector<int32_t> ind_, ind_t_;
  thrust::device_vector<int32_t> ptr_, ptr_t_;
  thrust::device_vector<T> val_, val_t_;

  std::vector<int32_t> host_ind_, host_ind_t_;
  std::vector<int32_t> host_ptr_, host_ptr_t_;
  std::vector<T> host_val_, host_val_t_;
};

#endif
