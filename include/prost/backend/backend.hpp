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

#ifndef PROST_BACKEND_HPP_
#define PROST_BACKEND_HPP_

#include <cmath>
#include "prost/common.hpp"
#include "prost/solver.hpp"

namespace prost {

template<typename T> class Problem;

///
/// \brief Abstract base class for primal-dual algorithms solving graph form 
///        problems.
/// 
template<typename T>
class Backend {
public:
  Backend() {}
  virtual ~Backend() {}

  void SetProblem(shared_ptr<Problem<T> > problem) { problem_ = problem; }
  void SetOptions(const typename Solver<T>::Options& opts) { solver_opts_ = opts; }

  virtual void Initialize() = 0;
  virtual void PerformIteration() = 0;
  virtual void Release() = 0;

  /// \brief Copies current primal dual solution pair (x,y) to the host.
  virtual void current_solution(vector<T>& primal_sol, vector<T>& dual_sol) = 0;

  /// \brief Copies current primal and dual solutions (x, z, y, w) to the host.
  virtual void current_solution(vector<T>& primal_x,
                                vector<T>& primal_z,
                                vector<T>& dual_y,
                                vector<T>& dual_w) = 0;

  /// \brief Returns norm of the primal residual |Ax - z|.
  virtual T primal_residual() const { return primal_residual_; }

  /// \brief Returns norm of the dual residual |A^T y + w|.
  virtual T dual_residual() const { return dual_residual_; }

  /// \brief Returns norm of the primal variable "z", used for stopping criterion.
  virtual T primal_var_norm() const  { return primal_var_norm_; }

  /// \brief Returns norm of the dual variable "w", used for stopping criterion.
  virtual T dual_var_norm() const { return dual_var_norm_; }

  /// \brief Returns primal stopping epsilon.
  virtual T eps_primal() const { return std::sqrt(problem_->nrows()) * solver_opts_.tol_abs_primal + solver_opts_.tol_rel_primal * primal_var_norm(); } 
  
  /// \brief Returns dual stopping epsilon.
  virtual T eps_dual() const { return std::sqrt(problem_->ncols()) * solver_opts_.tol_abs_dual + solver_opts_.tol_rel_dual * dual_var_norm(); } 

  // returns amount of gpu memory required in bytes
  virtual size_t gpu_mem_amount() const = 0;

protected:
  shared_ptr<Problem<T> > problem_;

  typename Solver<T>::Options solver_opts_;

  /// \brief Norm of the primal variable z = Kx.
  T primal_var_norm_;

  /// \brief Norm of the dual variable w = -K^T y.
  T dual_var_norm_;

  /// \brief Size of primal residual |Kx - z|.
  T primal_residual_;

  /// \brief Size of dual residual |K^T y + w|
  T dual_residual_;
};

} // namespace prost

#endif // PROST_BACKEND_HPP_
