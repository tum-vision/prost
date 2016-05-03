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

#ifndef PROST_PROX_IND_EPI_POLYHEDRAL_HPP_
#define PROST_PROX_IND_EPI_POLYHEDRAL_HPP_

#include "prost/common.hpp"
#include "prost/prox/prox_separable_sum.hpp"

namespace prost {

///
/// \brief Computes the orthogonal projection onto the 
///        epigraph of a piecewise linear function f described
///        by a maximum over linear functions
/// 
///        f(x) = max_{i=1..m} <a_i, x> + b_i
/// 
///        Here, the a_i are dim_ - 1 dimensional and the b_i are 1 
///        dimensional coefficients.
///
///        The prox solves the following optimization problem:
/// 
///        min_{x,t} (x - y)^2 + (t - u)^2
/// 
///        s.t. t >= <a_i, x> + b_i, i = 1..m
///
///        IMPORTANT: the a_i must be unique (no pointless constraints)
///                   and no more than DIM hyperplanes should intersect
///                   in one point (no more than DIM coplanar points in
///                   the dual representation).
/// 
template<typename T>
class ProxIndEpiPolyhedral : public ProxSeparableSum<T> {
public:
  /// \brief Constructor.
  /// 
  /// \param index     Start of the prox.
  /// \param count     Number of elements in the sum (e.g., number of pixels)
  /// \param dim       Dimension of the prox. f(x) is dim-1 dimensional.
  /// \param coeffs_a  A list of the m coefficients a_i, for every function, stored linearly.
  /// \param coeffs_b  A list of the m coefficients b_i, for every function, stored linearly.
  /// \param count_vec A list containing the number m for every function.
  /// \param index_vec A list containing the index into coeffs_a and coeffs_b for every function. 
  ///                  This is equal to the running sum over count_vec.
  ProxIndEpiPolyhedral(
    size_t index,
    size_t count,
    size_t dim, 
    bool interleaved,
    const vector<T>& coeffs_a,
    const vector<T>& coeffs_b, 
    const vector<uint32_t>& count_vec,
    const vector<uint32_t>& index_vec);

  virtual ~ProxIndEpiPolyhedral() {}

  virtual void Initialize();
  virtual size_t gpu_mem_amount() const;

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
  vector<T> host_coeffs_a_, host_coeffs_b_;
  vector<uint32_t> host_count_, host_index_;

  device_vector<T> dev_coeffs_a_, dev_coeffs_b_;
  device_vector<uint32_t> dev_count_, dev_index_;
};

} // namespace prost

#endif // PROST_PROX_IND_EPI_POLYHEDRAL_HPP_
