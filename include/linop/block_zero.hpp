#ifndef BLOCK_ZERO_HPP_
#define BLOCK_ZERO_HPP_

#include "block.hpp"
#include <cstdlib>
#include <vector>

/*
 * @brief Abstract base class for linear operator blocks.
 *
 */
namespace linop {

template<typename T>
class BlockZero : public Block<T> {
 public:
  BlockZero(size_t row, size_t col, size_t nrows, size_t ncols);


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
};
}
#endif
