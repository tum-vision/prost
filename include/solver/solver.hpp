#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <memory>
#include <vector>
#include <string>

class Problem;
class Backend;

template<typename T>
struct SolverOptions {
  // stopping tolerances
  T tol_rel_primal, tol_rel_dual;
  T tol_abs_primal, tol_abs_dual;

  int max_iters;

  // number of times the "intermediate solution" callback gets called
  int num_cback_calls;
};

enum SolverResult {
  CONVERGED,
  STOPPED_MAX_ITERS,
  STOPPED_USER,
};

typedef std::function<void(int, const std::vector<double>&, const std::vector<double>&, double, double)> IntermSolCallback;
typedef std::function<bool()> StoppingCallback;

/**
 * @brief General solver for graph form problems. 
 *
 */
template<typename T>
class Solver {
public:
  Solver();
  virtual ~Solver();

  void Initialize(Problem* problem, Backend* backend);
  SolverResult Solve();
  void Release();

  void SetOptions(const SolverOptions &opts);
  void SetStoppingCallback(const StoppingCallback& cb);
  void SetIntermSolCallback(const IntermSolCallback& cb);
  
protected:
  SolverOptions<T> opts;
  std::shared_ptr<Problem<T> > problem_;
  std::shared_ptr<Backend<T> > backend_;

  IntermSolCallback interm_sol_cback_;
  StoppingCallback stopping_cback_; // callback which allows for user input to stop the solver
};

#endif
