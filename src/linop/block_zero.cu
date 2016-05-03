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

#include "prost/linop/block_zero.hpp"

namespace prost {

template<typename T>
BlockZero<T>::BlockZero(size_t row, size_t col, size_t nrows, size_t ncols) 
  : Block<T>(row, col, nrows, ncols)
{
}

template<typename T>
BlockZero<T>::~BlockZero()
{
}

template<typename T>
void BlockZero<T>::EvalLocalAdd(
  const typename thrust::device_vector<T>::iterator& res_begin,
  const typename thrust::device_vector<T>::iterator& res_end,
  const typename thrust::device_vector<T>::const_iterator& rhs_begin,
  const typename thrust::device_vector<T>::const_iterator& rhs_end)
{
  // do nothing for zero operator
}

template<typename T>
void BlockZero<T>::EvalAdjointLocalAdd(
  const typename thrust::device_vector<T>::iterator& res_begin,
  const typename thrust::device_vector<T>::iterator& res_end,
  const typename thrust::device_vector<T>::const_iterator& rhs_begin,
  const typename thrust::device_vector<T>::const_iterator& rhs_end)
{
  // do nothing for zero operator
}

// Explicit template instantiation
template class BlockZero<float>;
template class BlockZero<double>;

} // namespace prost
