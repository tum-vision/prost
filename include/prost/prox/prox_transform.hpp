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

#ifndef PROST_PROX_TRANSFORM_HPP_
#define PROST_PROX_TRANSFORM_HPP_

#include "prost/prox/prox.hpp"

namespace prost {

/// 
/// \brief Computes the proximal operator of the transformed function
///        h(x) = c f(ax - b) + <d, x> + (e/2) <x, x>,
///        using the proximal operator of f. 
/// 
template<typename T>
class ProxTransform : public Prox<T> {
public:
  ProxTransform(
    shared_ptr<Prox<T> > inner_fn,
    const vector<T>& a,
    const vector<T>& b,
    const vector<T>& c,
    const vector<T>& d,
    const vector<T>& e);

  virtual ~ProxTransform() { }

  virtual void Initialize();
  virtual void Release();

  virtual size_t gpu_mem_amount() const;
  virtual void get_separable_structure(vector<std::tuple<size_t, size_t, size_t> >& sep);

protected:
  virtual void EvalLocal(
    const typename device_vector<T>::iterator& result_beg,
    const typename device_vector<T>::iterator& result_end,
    const typename device_vector<T>::const_iterator& arg_beg,
    const typename device_vector<T>::const_iterator& arg_end,
    const typename device_vector<T>::const_iterator& tau_beg,
    const typename device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);

private:
  shared_ptr<Prox<T> > inner_fn_;
  device_vector<T> scaled_arg_;
  device_vector<T> scaled_tau_;

  vector<T> host_a_, host_b_, host_c_, host_d_, host_e_;
  device_vector<T> dev_a_, dev_b_, dev_c_, dev_d_, dev_e_;
};


} // namespace prost

#endif // PROST_PROX_TRANSFORM_HPP_
