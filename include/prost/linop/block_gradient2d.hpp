#ifndef PROST_BLOCK_GRADIENT2D_HPP_
#define PROST_BLOCK_GRADIENT2D_HPP_

#include "prost/linop/block.hpp"

namespace prost {

///
/// \brief Computes the gradient in a 2D domain with Forward-Differences
///        and Neumann boundary conditions.
///        Assumes pixel-first (matlab style, rows first) ordering and
///        outputs dx dy pixelwise ordered.
///
///        If label_first == true, assumes label first then rows then cols 
///        ordering but still outputs dx dy pixelwise ordered.
///
template<typename T>
class BlockGradient2D : public Block<T>
{
  BlockGradient2D(size_t row,
		  size_t col,
		  size_t nrows,
		  size_t ncols,
		  size_t nx,
		  size_t ny,
		  size_t L,
		  bool label_first);
  
  virtual ~BlockGradient2D() {}

  virtual size_t gpu_mem_amount() const { return 0; }
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
  size_t nx_;
  size_t ny_;
  size_t L_;
  bool label_first_;
};

}

#endif // PROST_BLOCK_GRADIENT2D_HPP_
  
