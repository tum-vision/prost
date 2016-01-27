#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <memory>
#include <vector>
#include <string>

template<typename T> class Problem;
template<typename T> class Backend;

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
    kConverged,
    kStoppedMaxIters,
    kStoppedUser
  };

  /// \brief Intermediate solution callback. Arguments: (iteration, primal_solution, dual_solution). 
  typedef std::function<void(int, const std::vector<T>&, const std::vector<T>&)> IntermCallback;

  /// \brief Stopping callback. Used to terminate the solver
  ///        prematurely (i.e. by user input from Matlab).
  typedef std::function<bool()> StoppingCallback;
  
  Solver(std::shared_ptr<Problem<T> > problem, std::shared_ptr<Backend<T> > backend);
  virtual ~Solver();

  void Initialize();
  typename Solver<T>::ConvergenceResult Solve();
  void Release();

  void SetOptions(const typename Solver<T>::Options &opts);
  void SetStoppingCallback(const typename Solver<T>::StoppingCallback& cb);
  void SetIntermCallback(const typename Solver<T>::IntermCallback& cb);

  const std::vector<T>& cur_primal_sol() const { return cur_primal_sol_; }
  const std::vector<T>& cur_dual_sol() const { return cur_dual_sol_; }
  
protected:
  Solver<T>::Options opts_;
  std::shared_ptr<Problem<T> > problem_;
  std::shared_ptr<Backend<T> > backend_;

  std::vector<T> cur_primal_sol_;
  std::vector<T> cur_dual_sol_;

  typename Solver<T>::IntermCallback interm_cb_;
  typename Solver<T>::StoppingCallback stopping_cb_;
};

#endif
