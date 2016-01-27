#ifndef BLOCK_SPARSE_HPP_
#define BLOCK_SPARSE_HPP_

#include "block.hpp"
#include "util/sparse_matrix.hpp"
#include "../pdsolver_exception.hpp"

/**
 * @brief Linear Operator represented as a sparse matrix. Takes ownership
 *        of the pointer to the SparseMatrix.
 */
namespace linop {
template<typename T>
class BlockSparse : public Block<T> {
 public:
  BlockSparse(size_t row,
              size_t col,
              std::unique_ptr<SparseMatrix<T>> mat);


  virtual size_t gpu_mem_amount() const;

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;
  
 protected:
  virtual void EvalLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end);
  virtual void EvalAdjointLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end);
  
  std::unique_ptr<SparseMatrix<T>> mat_;
};
}
#endif