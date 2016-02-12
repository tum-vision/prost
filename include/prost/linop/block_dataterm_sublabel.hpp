#ifndef PROST_BLOCK_DATATERM_SUBLABEL_HPP_
#define PROST_BLOCK_DATATERM_SUBLABEL_HPP_

#include "prost/linop/block.hpp"

namespace prost {

/// 
/// \brief Implementation of a part of the linear constraints given for
///        the saddle-point problem in the paper:
///        http://arxiv.org/abs/1512.01383
///
template<typename T>
class BlockDatatermSublabel : public Block<T> {
public:
  BlockDatatermSublabel(size_t row, size_t col, size_t nx, size_t ny, size_t L, T left, T right);
  virtual ~BlockDatatermSublabel() { }

  virtual size_t gpu_mem_amount() const { return 0; }

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const; 
  virtual T col_sum(size_t col, T alpha) const;

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

private:
  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
  
  T t_min_;
  T t_max_;
};

}

#endif // PROST_BLOCK_DATATERM_SUBLABEL_HPP_
