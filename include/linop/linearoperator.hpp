#ifndef LINEAROPERATOR_HPP_
#define LINEAROPERATOR_HPP_

#include <cstdlib>
#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include "block.hpp"

/*
 * @brief Linear operator built out of blocks.
 *
 */
template<typename T>
class LinearOperator {
 public:
  LinearOperator();
  virtual ~LinearOperator();

  void AddBlock(std::shared_ptr<Block<T> > block);
  
  void Initialize();
  void Release();

  void Eval(
    thrust::device_vector<T>& result, 
    const thrust::device_vector<T>& rhs);

  void EvalAdjoint(
    thrust::device_vector<T>& result, 
    const thrust::device_vector<T>& rhs);

  // returns \sum_{col=1}^{ncols} |K_{row,col}|^{\alpha}
  T row_sum(size_t row, T alpha) const;

  // returns \sum_{row=1}^{nrows} |K_{row,col}|^{\alpha}
  T col_sum(size_t col, T alpha) const;

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; } 

  size_t gpu_mem_amount() const;
  
 protected:
  std::vector<std::shared_ptr<Block<T> > > blocks_;
  size_t nrows_;
  size_t ncols_;
};

#endif
