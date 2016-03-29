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
  typedef function<void(int, const vector<T>&, const vector<T>&)> IntermCallback;

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
