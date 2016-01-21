/**
#ifndef BLOCK_HPP_
#define BLOCK_HPP_

#include <cstdlib>
#include <vector>

/*
 * @brief Abstract base class for linear operator blocks.
 *
 */
namespace linop {
template<typename T>
class Block {
 public:
  Block(size_t row, size_t col, size_t nrows, size_t ncols);
  virtual ~Block();

  virtual bool Init();
  virtual void Release();
  
  void EvalAdd(T *d_res, T *d_rhs);
  void EvalAdjointAdd(T *d_res, T *d_rhs);

  // required for preconditioners
  // row and col are "local" for the operator, which means they start at 0
  virtual T row_sum(size_t row, T alpha) const { return 0; }
  virtual T col_sum(size_t col, T alpha) const { return 0; }

  size_t row() const { return row_; }
  size_t col() const { return col_; }
  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  virtual size_t gpu_mem_amount() const { return 0; }
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs) = 0;
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs) = 0;
  
  size_t row_;
  size_t col_;
  size_t nrows_;
  size_t ncols_;
};
}
#endif
*/