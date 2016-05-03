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

#ifndef PROST_SHARED_MEM_HPP_
#define PROST_SHARED_MEM_HPP_

namespace prost {

// TODO: comment me
template<typename T, class F>
class SharedMem 
{
public:
  __device__
  SharedMem(size_t dim, size_t threadIdx_x)
      : dim_(dim), threadIdx_x_(threadIdx_x)
  {
    extern __shared__ char sh_mem[];
    sh_arg_ = reinterpret_cast<T*>(sh_mem);
  }

  inline __device__
  T operator[](size_t i) const
  {
    size_t index = threadIdx_x_ * get_count_fun_(dim_) + i;
    return sh_arg_[index];
  }

  inline __device__
  T& operator[](size_t i)
  {
    // Out of bounds check?
    size_t index = threadIdx_x_ * get_count_fun_(dim_) + i;
    return sh_arg_[index];
  }

private:
  size_t dim_;
  size_t threadIdx_x_;
  T* sh_arg_;
  F get_count_fun_;
};

} // namespace prost

#endif // PROST_SHARED_MEM_HPP_
