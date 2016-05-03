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

#ifndef PROST_VECTOR_HPP_
#define PROST_VECTOR_HPP_

namespace prost {
///
/// \brief Helper class for proximal operators which abstracts away the
///        different access patterns to global memory / the variables.
///      
template<typename T>
//TODO rename deviceVectorView
class Vector
{
public:
  __host__ __device__
  Vector(size_t count, size_t dim, bool interleaved, size_t tx, T* data) :
    count_(count),
    dim_(dim),
    interleaved_(interleaved),
    tx_(tx),
    data_(data) { }

  inline __host__ __device__
  T&
  operator[](size_t i) const
  {
    size_t index = interleaved_ ? (tx_ * dim_ + i) : (tx_ + count_ * i);
    return data_[index];
  }
  
private:
  size_t count_;
  size_t dim_;
  bool interleaved_;
  size_t tx_;
  T* data_;
};

} // namespace prost

#endif // PROST_VECTOR_HPP_
