#ifndef SOLVER_BACKEND_PDHGTINY_HPP_
#define SOLVER_BACKEND_PDHGTINY_HPP_

#include "solver_backend.hpp"

/**
 * @brief ...
 */
class SolverBackendPDHGTiny : public SolverBackend {
public:
  SolverBackendPDHGTiny(
      const OptimizationProblem& problem,
      const SolverOptions& opts)
      : SolverBackend(problem, opts) { }
  virtual ~SolverBackendPDHGTiny() { }
  
  virtual bool Initialize();
  virtual void PerformIteration();
  virtual void Release();

  virtual void iterates(real *primal, real *dual);
  virtual bool converged();
  virtual std::string status();

  // returns amount of gpu memory required in bytes
  virtual int gpu_mem_amount();
  
protected:
  // algorithm variables
  real *d_x_; // primal variable x^k
  real *d_y_; // dual variable y^k
  real *d_x_bar_; // overrelaxed variable
  real *d_temp_; // temporary variable for all sorts of stuff

  real tau_; // primal step size
  real sigma_; // dual step size
};

#endif
