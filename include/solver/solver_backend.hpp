#ifndef SOLVER_BACKEND_HPP_
#define SOLVER_BACKEND_HPP_

#include <vector>

#include "config.hpp"
#include "prox/prox.hpp"
#include "solver/solver.hpp"
#include "util/sparse_matrix.hpp"

/**
 * @brief ...
 *
 */
class SolverBackend {
public:
  SolverBackend(const OptimizationProblem& problem, const SolverOptions& opts)
      : problem_(problem), opts_(opts) { }
  virtual ~SolverBackend() { }

  virtual bool Initialize() = 0;
  virtual void PerformIteration() = 0;
  virtual void Release() = 0;

  virtual void iterates(real *primal, real *dual) = 0;
  virtual bool converged() = 0;
  virtual std::string status() = 0;

  // returns amount of gpu memory required in bytes
  virtual int gpu_mem_amount() = 0;

protected:
  OptimizationProblem problem_;
  SolverOptions opts_;
  int iteration_;
};

#endif
