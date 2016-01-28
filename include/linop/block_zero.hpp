#ifndef BLOCK_ZERO_HPP_
#define BLOCK_ZERO_HPP_

#include <cstdlib>
#include <vector>

#include "linop/block.hpp"

/*
 * @brief Zero linear operator
 *
 */
template<typename T>
class BlockZero : public Block<T> 
{
public:
  BlockZero(size_t row, size_t col, size_t nrows, size_t ncols);
  virtual ~BlockZero();

  virtual T row_sum(size_t row, T alpha) const { return 0; }
  virtual T col_sum(size_t col, T alpha) const { return 0; }

  virtual size_t gpu_mem_amount() const { return 0; }

protected:
  virtual void EvalLocalAdd(
    const thrust::device_ptr<T>& result, 
    const thrust::device_ptr<const T>& rhs);

  virtual void EvalAdjointLocalAdd(
    const thrust::device_ptr<T>& result, 
    const thrust::device_ptr<const T>& rhs);
};

#endif
