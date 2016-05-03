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
