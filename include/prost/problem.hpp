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

#ifndef PROST_PROBLEM_HPP_
#define PROST_PROBLEM_HPP_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "prost/common.hpp"

namespace prost {

using thrust::device_vector;
using thrust::device_ptr;

template<typename T> class Prox;
template<typename T> class Block;
template<typename T> class LinearOperator;
template<typename T> class DualLinearOperator;

/// @brief Contains all information describing the graph form problem
/// 
///         min_{x,z} g(\Tau^{1/2} x) + f(\Sigma^{-1/2} z) 
/// 
///         s.t. z = \Sigma^{1/2} K \Tau^{1/2} x. 
/// 
///         scaling_left_host_ = Sigma,
///         scaling_right_host_ = Tau.
/// 
template<typename T>
class Problem {
public:
  enum Scaling {
    /// \brief No preconditioning.
    kScalingIdentity,

    /// \brief Pock, Chambolle ICCV '11 preconditioning.
    kScalingAlpha, 

    /// \brief User defined preconditioning.
    kScalingCustom,
  };

  typedef vector<shared_ptr<Prox<T>>> ProxList;

  Problem();
  virtual ~Problem() {}

  void AddBlock(shared_ptr<Block<T> > block);
  void AddProx_g(shared_ptr<Prox<T> > prox);
  void AddProx_f(shared_ptr<Prox<T> > prox);
  void AddProx_gstar(shared_ptr<Prox<T> > prox);
  void AddProx_fstar(shared_ptr<Prox<T> > prox);

  /// \brief Builds the linear operator and checks if prox cover the 
  ///        whole domain.
  void Initialize();
  void Release();

  /// \brief Sets a predefined problem scaling.
  /// \param[in] left Left scaling/preconditioner diagonal matrix Sigma^{1/2}.
  /// \param[in] right Right scaling/preconditioner diagonal matrix Tau^{1/2}.
  void SetScalingCustom(
    const vector<T>& left, 
    const vector<T>& right);

  /// \brief Set the scaling to the Diagonal Preconditioners
  ///        proposed in Pock, Chambolle ICCV '11. 
  void SetScalingAlpha(T alpha);

  /// \brief Set the preconditioners to Sigma = Tau = Identity.
  void SetScalingIdentity();

  /// \brief
  void SetDimensions(size_t nrows, size_t ncols) { nrows_ = nrows; ncols_ = ncols; }

  shared_ptr<LinearOperator<T>> linop() const { return linop_; }
  device_vector<T>& scaling_left() { return scaling_left_; }
  device_vector<T>& scaling_right() { return scaling_right_; }
  const ProxList& prox_f() const { return prox_f_; }
  const ProxList& prox_g() const { return prox_g_; }
  const ProxList& prox_fstar() const { return prox_fstar_; }
  const ProxList& prox_gstar() const { return prox_gstar_; }

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  size_t gpu_mem_amount() const;

  /// \brief Estimates the norm of the scaled linear operator via power iteration.
  T normest(T tol = 1e-6, int max_iters = 100);

  /// \brief Dualizes the problem by doing the following swappings:
  ///        Swap g <-> f*, f <-> g*, K <-> -K^T
  ///        Should be called after Initialize(). 
  void Dualize();
  
protected:
  size_t nrows_, ncols_; // problem size

  /// \brief matrix K
  shared_ptr<LinearOperator<T>> linop_;

  /// \brief matrix -K^T
  shared_ptr<LinearOperator<T>> dual_linop_;

  typename Problem<T>::Scaling scaling_type_;

  /// \brief (squared) Left-preconditioner Sigma
  device_vector<T> scaling_left_; 

  /// \brief (squared) Right-preconditioner Tau
  device_vector<T> scaling_right_;

  /// \brief User-defined (squared) left-preconditioner Sigma
  vector<T> scaling_left_host_;

  /// \brief User-defined (squared) right-preconditioner Tau
  vector<T> scaling_right_host_;

  /// \brief alpha for Pock-preconditioning
  T scaling_alpha_;

  /// \brief proximal operators (may be overcomplete).
  ProxList prox_f_;
  ProxList prox_g_;
  ProxList prox_fstar_;
  ProxList prox_gstar_;

private:
  /// \brief Averages the values for the preconditioner at the entries where
  ///        prox does not allow diagonal step sizes.
  void AveragePreconditioners(
    vector<T>& precond,
    const ProxList& prox);
};

} // namespace prost

#endif // PROST_PROBLEM_HPP_
