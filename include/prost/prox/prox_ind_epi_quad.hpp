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

#ifndef PROST_PROX_IND_EPI_QUAD_
#define PROST_PROX_IND_EPI_QUAD_

#include <array>
#include <vector>
#include <thrust/device_vector.h>

#include "prost/prox/prox_separable_sum.hpp"
#include "prost/prox/vector.hpp"
#include "prost/common.hpp"

namespace prost {

///
/// \brief Implements projection onto epigraph of quadratic function
///        min_{x,y} ||(x,y) - (u,v)|| s.t. a x^T x + b^T x + c <= y
///
template<typename T>
class ProxIndEpiQuad : public ProxSeparableSum<T> 
{
public:    
  ProxIndEpiQuad(
      size_t index, 
      size_t count, 
      size_t dim, 
      bool interleaved, 
      bool diagsteps, 
      std::vector<T> a,
      std::vector<T> b,
      std::vector<T> c)
      : ProxSeparableSum<T>(index, count, dim, interleaved, diagsteps), a_(a), b_(b), c_(c) { }

  virtual void Initialize();
  
  virtual size_t gpu_mem_amount() const
  {
    return (this->count_*this->dim_ + this->count_) *sizeof(T);
  }
   
protected:
  virtual void EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
  
private:    
  thrust::device_vector<T> d_a_;
  thrust::device_vector<T> d_b_;
  thrust::device_vector<T> d_c_;  
 
  std::vector<T> a_;
  std::vector<T> b_;
  std::vector<T> c_;
};

} // namespace prost

#endif // PROST_PROJ_EPI_QUAD_
