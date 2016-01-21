/**
#ifndef BLOCK_DIAGS_HPP_
#define BLOCK_DIAGS_HPP_

#include "block.hpp"

/**
 * @brief Linear operator implementation of the MATLAB command speye.
 *
 * @param ndiags: number of diagonals
 * @param offsets: array of size ndiags, starting position of diagonals
 * @param factors: array of size ndiags, constant factor each diagonal
 *                 is multiplied with
 */
namespace linop {
template<typename T>
class BlockDiags : public Block<T> {
 public:
  BlockDiags(size_t row,
             size_t col,
             size_t nrows,
             size_t ncols,
             size_t ndiags,
             const std::vector<ssize_t>& offsets,
             const std::vector<T>& factors);
  
  virtual ~BlockDiags();

  virtual bool Init();
  virtual void Release();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;
  
  static void ResetConstMem() { cmem_counter_ = 0; }

 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t cmem_offset_;
  size_t ndiags_;
  std::vector<ssize_t> offsets_;
  std::vector<float> factors_;

  static size_t cmem_counter_;
};
}
#endif
*/