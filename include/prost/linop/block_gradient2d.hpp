/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

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
public:
  BlockGradient2D(size_t row,
		  size_t col,
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
