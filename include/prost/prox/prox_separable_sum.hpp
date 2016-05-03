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

#ifndef PROST_PROX_SEPARABLE_SUM_HPP_
#define PROST_PROX_SEPARABLE_SUM_HPP_

#include "prost/prox/prox.hpp"

namespace prost {

/// 
/// \brief Abstract base class for proximal operators 
///        for a sum of separable functions:
/// 
///        sum_{i=index_}^{index_+count_} f_i(x_i),
/// 
///        where the f_i and x_i are dim_ dimensional.
/// 
///        interleaved_ describes the ordering of the elements if dim_ > 1.
///        If it is set to true, then successive elements in x correspond
///        to one of count_ many dim_-dimensional vectors.
///        If interleaved_ is set of false, then there are dim_ contigiuous
///        chunks of count_ many elements.
/// 
///        Example: gradient operator for 3d image with 5 pixels
///        interleaved_ == true : dx dy dz dx dy dz dx dy dz dx dy dz dx dy dz
///        interleaved_ == false: dx dx dx dx dx dy dy dy dy dy dz dz dz dz dz
/// 
template<typename T>
class ProxSeparableSum : public Prox<T> 
{
public:
  ProxSeparableSum(size_t index, 
                   size_t count, 
                   size_t dim, 
                   bool interleaved,
                   bool diagsteps)
      
      : Prox<T>(index, count*dim, diagsteps),
      count_(count),
      dim_(dim),
      interleaved_(interleaved) { }

  size_t dim() const { return dim_; }
  size_t count() const { return count_; }
  bool interleaved() const { return interleaved_; }

  virtual void get_separable_structure(vector<std::tuple<size_t, size_t, size_t> >& sep) {
    if(interleaved()) {
      for(size_t i = 0; i < count(); i++)
        sep.push_back( 
          std::tuple<size_t, size_t, size_t>(this->index() + i * dim(), dim(), 1) );
    }
    else
    {
      for(size_t i = 0; i < count(); i++)
        sep.push_back( 
          std::tuple<size_t, size_t, size_t>(this->index() + i, dim(), count()) );
    }
  }


protected:
  size_t count_; 
  size_t dim_;

  /// \brief Ordering of elements if dim > 1.
  bool interleaved_; 
};

} // namespace prost

#endif // PROST_PROX_SEPARABLE_SUM_HPP_

