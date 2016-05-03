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

#ifndef PROST_SOLVER_HPP_
#define PROST_SOLVER_HPP_

#include "prost/common.hpp"

namespace prost {

template<typename T> class Problem;
template<typename T> class Backend;

/// 
/// \brief Solver for graph-form problems.
///
/// @tparam typename T. Floating point-type.
/// 
template<typename T>
class Solver {
public:
  struct Options {
    /// \brief relative primal stopping tolerance
    T tol_rel_primal;

    /// \brief relative dual stopping tolerance
    T tol_rel_dual;

    /// \brief absolute primal stopping tolerance
    T tol_abs_primal;

    /// \brief absolute dual stopping tolerace
    T tol_abs_dual;

    /// \brief maximum number of iterations
    int max_iters;

    /// \brief total number of times the "intermediate solution" callback should 
    ///        be called within the specified number of maximum iterations.
    int num_cback_calls;

    /// \brief output various debug information
    bool verbose;

    /// \brief Initial primal solution
    vector<T> x0;

    /// \brief Initial dual solution
    vector<T> y0;

    /// \brief Solve the dual or primal problem?
    bool solve_dual_problem;
  };

  enum ConvergenceResult {
    kConverged,
    kStoppedMaxIters,
    kStoppedUser
  };

  /// \brief Intermediate solution callback. Arguments: (iteration, primal_solution, dual_solution). 
  typedef function<bool(int, const vector<T>&, const vector<T>&)> IntermCallback;

  /// \brief Stopping callback. Used to terminate the solver
  ///        prematurely (i.e. by user input from Matlab).
  typedef function<bool()> StoppingCallback;
  
  Solver(shared_ptr<Problem<T>> problem, shared_ptr<Backend<T>> backend);
  virtual ~Solver() {}

  void Initialize();
  typename Solver<T>::ConvergenceResult Solve();
  void Release();

  void SetOptions(const typename Solver<T>::Options &opts);
  void SetStoppingCallback(const typename Solver<T>::StoppingCallback& cb);
  void SetIntermCallback(const typename Solver<T>::IntermCallback& cb);

  const vector<T>& cur_primal_sol() const; 
  const vector<T>& cur_dual_sol() const;
  const vector<T>& cur_primal_constr_sol() const;
  const vector<T>& cur_dual_constr_sol() const;
  
protected:
  typename Solver<T>::Options opts_;
  shared_ptr<Problem<T>> problem_;
  shared_ptr<Backend<T>> backend_;

  vector<T> cur_primal_sol_; // x
  vector<T> cur_dual_sol_; // y
  vector<T> cur_primal_constr_sol_; // z
  vector<T> cur_dual_constr_sol_; // w

  typename Solver<T>::IntermCallback interm_cb_;
  typename Solver<T>::StoppingCallback stopping_cb_;
};

} // namespace prost

#endif // PROST_SOLVER_HPP_
