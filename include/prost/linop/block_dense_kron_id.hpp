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

#ifndef PROST_BLOCK_DENSE_KRON_ID_HPP_
#define PROST_BLOCK_DENSE_KRON_ID_HPP_

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "prost/linop/block.hpp"

namespace prost {

/// 
/// \brief Linear operator based on dense matrix.
/// 
template<typename T>
class BlockDenseKronId : public Block<T> {
};

} // namespace prost

#endif // PROST_BLOCK_DENSE_KRON_ID_HPP_
