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

#ifndef PROST_ELEM_OPERATION_HPP_
#define PROST_ELEM_OPERATION_HPP_

#include "prost/prox/shared_mem.hpp"
#include "prost/prox/vector.hpp"

namespace prost {

// TODO: comment me
template<size_t DIM = 0, size_t COEFFS_COUNT = 0, typename SHARED_MEM_TYPE = char>
struct ElemOperation 
{
public: 
  static const size_t kCoeffsCount = COEFFS_COUNT;
  static const size_t kDim = DIM;
  struct GetSharedMemCount {
    inline __host__ __device__ size_t operator()(size_t dim) { return 0; }
  };
  typedef SHARED_MEM_TYPE SharedMemType;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_HPP_
