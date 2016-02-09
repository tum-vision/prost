#ifndef PROST_BLOCK_ZERO_HPP_
#define PROST_BLOCK_ZERO_HPP_

#include "prost/linop/block.hpp"

namespace prost {

/// \brief Zero operator.
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
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

  virtual void EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);
};

} // namespace prost

#endif // PROST_BLOCK_ZERO_HPP
