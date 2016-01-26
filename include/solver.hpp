#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <memory>
#include <vector>
#include <string>

class Problem;
class Backend;

/// 
/// \brief Solver for graph-form problems.
///
/// @tparam typename T. Floating point-type
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
  };

  enum ConvergenceResult {
    CONVERGED, 
    STOPPED_MAX_ITERS,
    STOPPED_USER,
  };

  /// \brief Intermediate solution callback. Arguments: (iteration, primal_solution, dual_solution). 
  typedef std::function<void(int, const std::vector<double>&, const std::vector<double>&)> IntermCallback;

  /// \brief Stopping callback. Returns true if the solver should be terminated 
  ///        prematurely (i.e. by user input from Matlab).
  typedef std::function<bool()> StoppingCallback;
  
  Solver(std::shared_ptr<Problem> problem, std::shared_ptr<Backend> backend);
  virtual ~Solver();

  void Initialize();
  typename Solver<T>::ConvergenceResult Solve();
  void Release();

  void SetOptions(const SolverOptions &opts);
  void SetStoppingCallback(const StoppingCallback& cb);
  void SetIntermCallback(const IntermCallback& cb);
  
protected:
  Solver<T>::Options opts_;
  std::shared_ptr<Problem<T> > problem_;
  std::shared_ptr<Backend<T> > backend_;

  std::vector<T> cur_primal_sol_;
  std::vector<T> cur_dual_sol_;

  IntermSolCallback interm_cb_;
  StoppingCallback stopping_cb_; // callback which allows for user input to stop the solver
};

#endif
