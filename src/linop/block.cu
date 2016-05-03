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

#include "prost/linop/block.hpp"

namespace prost {

template<typename T>
Block<T>::Block(size_t row, size_t col, size_t nrows, size_t ncols) 
  : row_(row), col_(col), nrows_(nrows), ncols_(ncols) 
{
}

template<typename T>
Block<T>::~Block() 
{ 
}

template<typename T>
void Block<T>::Initialize()
{
}

template<typename T>
void Block<T>::Release()
{
}
  
template<typename T>
void Block<T>::EvalAdd(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& rhs)
{
  EvalLocalAdd(
    result.begin() + row_,
    result.begin() + row_ + nrows_,
    rhs.cbegin() + col_,
    rhs.cbegin() + col_ + ncols_);
}

template<typename T>
void Block<T>::EvalAdjointAdd(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& rhs)
{
  EvalAdjointLocalAdd(
    result.begin() + col_,
    result.begin() + col_ + ncols_,
    rhs.cbegin() + row_,
    rhs.cbegin() + row_ + nrows_);
}

// Explicit template instantiation
template class Block<float>;
template class Block<double>;

} // namespace prost
