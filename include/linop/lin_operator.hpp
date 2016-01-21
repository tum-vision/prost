/**
#ifndef LIN_OPERATOR_HPP_
#define LIN_OPERATOR_HPP_

#include <cstdlib>
#include <vector>

#include "block.hpp"

/*
 * @brief Block matrix built out of linops.
 *
 */

namespace linop {

template<typename T>
class LinOperator {
 public:
  LinOperator();
  virtual ~LinOperator();

  // careful: transfers ownership to LinearOperator
  void AddBlock(shared_ptr<Block<T>> op);
  
  bool Init();
  void Release();

  void Eval(T *d_res, T *d_rhs);
  void EvalAdjoint(T *d_res, T *d_rhs);

  // required for preconditioners
  T row_sum(size_t row, T alpha) const;
  T col_sum(size_t col, T alpha) const;

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; } 

  size_t gpu_mem_amount() const;
  
 protected:
  std::vector<shared_ptr<Block<T>>> blocks_;
  size_t nrows_;
  size_t ncols_;
};
}
#endif
*/