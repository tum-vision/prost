#ifndef LIN_OPERATOR_HPP_
#define LIN_OPERATOR_HPP_

#include <cstdlib>
#include <vector>
#include <thrust/device_vector.h>



#include "block.hpp"
#include "../pdsolver_exception.hpp"

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
  void AddBlock(std::unique_ptr<Block<T>> op);
  
  void Init();
  void Release();

  void Eval(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs);
  void EvalAdjoint(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs);

  // required for preconditioners
  T row_sum(size_t row, T alpha) const;
  T col_sum(size_t col, T alpha) const;

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; } 

  size_t gpu_mem_amount() const;
  
 private:
   bool RectangleOverlap(size_t x1, size_t y1,
                      size_t x2, size_t y2,
                      size_t a1, size_t b1,
                      size_t a2, size_t b2);

  std::vector<std::unique_ptr<Block<T>>> blocks_;
  size_t nrows_;
  size_t ncols_;
};
}
#endif