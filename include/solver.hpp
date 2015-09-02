#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <memory>
#include <vector>

#include "config.hpp"
#include "preconditioner.hpp"
#include "prox.hpp"
#include "util/sparse_matrix.hpp"

typedef void (*SolverCallbackPtr)(int, real*, real*, bool);

enum SolverBackendType
{
  kBackendPDHG,
};

enum BackendPDHGType
{
  kPDHGAlg1,
  kPDHGAlg2,
  kPDHGAdapt,
};

struct SolverOptions {
  // default options
  SolverOptions() {
    backend = kBackendPDHG;
    max_iters = 1000;
    cb_iters = 10;
    tolerance = 1e-6;
    verbose = true;
    
    precond = kPrecondScalar;
    precond_alpha = 1;
    
    pdhg = kPDHGAlg1;
    gamma = 0;
    alpha0 = 0;
    nu = 0;
    delta = 0;
    s = 0;
  }

  std::string get_string() const;
  
  SolverBackendType backend;

  // generic solver options
  int max_iters;
  int cb_iters;
  real tolerance;
  bool verbose;

  // parameters for preconditioner
  PreconditionerType precond;
  real precond_alpha;

  // parameters for pdhg
  BackendPDHGType pdhg;
  real gamma;
  real alpha0, nu, delta, s;  
};

struct OptimizationProblem {
  OptimizationProblem()
      : mat(NULL), precond(NULL)
  {
  }
  
  int nrows, ncols;
  std::vector<Prox*> prox_g;
  std::vector<Prox*> prox_hc;
  SparseMatrix<real> *mat;
  Preconditioner *precond;
};

// forward declaration because solver_backend.hpp needs to include this file.
class SolverBackend;

/**
 * @brief ...
 *
 */
class Solver {
public:
  Solver();
  virtual ~Solver();

  // Solver takes ownership of mat
  void SetMatrix(SparseMatrix<real>* mat);

  // Solver takes ownership of prox
  void SetProx_g(const std::vector<Prox*>& prox);
  void SetProx_hc(const std::vector<Prox*>& prox);
  
  void SetOptions(const SolverOptions& opts);
  void SetCallback(SolverCallbackPtr cb); 

  bool Initialize();
  void Solve();
  void Release();

  real *primal_iterate() const { return h_primal_; }
  real *dual_iterate() const { return h_dual_; }
  
protected:
  SolverBackend *backend_;
  SolverOptions opts_;
  real *h_primal_;
  real *h_dual_;
  SolverCallbackPtr callback_;
  OptimizationProblem problem_;
};

#endif
