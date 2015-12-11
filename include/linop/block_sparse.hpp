#ifndef BLOCK_SPARSE_HPP_
#define BLOCK_SPARSE_HPP_

#include "block.hpp"
#include "util/sparse_matrix.hpp"

/**
 * @brief Linear Operator represented as a sparse matrix. Takes ownership
 *        of the pointer to the SparseMatrix.
 */
namespace linop {
template<typename T>
class BlockSparse : public LinOp<T> {
 public:
  BlockSparse(size_t row,
              size_t col,
              SparseMatrix<T> *mat);

  virtual ~BlockSparse();

  virtual bool Init();
  virtual void Release();

  virtual size_t gpu_mem_amount() const { return mat_->gpu_mem_amount(); }

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);
  
  SparseMatrix<T> *mat_;
};
}
#endif
