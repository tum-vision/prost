#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <memory>
#include <vector>
#include <string>

#include "config.hpp"
#include "preconditioner.hpp"
#include "prox/prox.hpp"
#include "util/sparse_matrix.hpp"

typedef void (*SolverCallbackPtr)(int, real*, real*, bool);

enum SolverBackendType
{
  kBackendPDHG,
};

enum AdaptivityType {
  kAdaptNone,           // constant step sizes
  kAdaptStrong, // adapts step sizes based on strong convexity
  kAdaptBalance,        // tries to balance residuals
  kAdaptConverge,       // lets one of the residuals converge first, then adapts.
};

struct SolverOptions {
  // default options
  SolverOptions() {
    backend = kBackendPDHG;
    max_iters = 1000;
    cb_iters = 10;
    tol_primal = 1.0;
    tol_dual = 1.0;
    verbose = true;

    // preconditioning
    precond = kPrecondScalar;
    precond_alpha = 1;

    // adaptivity
    adapt = kAdaptNone;
    ad_strong.gamma = 0;
    ad_balance.alpha0 = 0.5;
    ad_balance.nu = 0.95;
    ad_balance.delta = 1.5;
    ad_balance.s = 10;
    ad_converge.delta = 1.05;
    ad_converge.tau = 0.8;

    // backtracking
    bt_enabled = false;
    bt_beta = 0.95;
    bt_gamma = 0.75;
  }

  std::string get_string() const;
  
  SolverBackendType backend;

  // generic solver options
  int max_iters;
  int cb_iters;
  real tol_primal;
  real tol_dual;
  bool verbose;

  // parameters for preconditioner
  PreconditionerType precond;
  real precond_alpha;

  // adaptivity parameters for pdhg
  struct AdaptParamsStrong {
    real gamma;
  };

  struct AdaptParamsBalance {
    real alpha0;
    real nu;
    real delta;
    real s;
  };

  struct AdaptParamsConverge {
    real delta;
    real tau;
  };
  
  AdaptivityType adapt;
  AdaptParamsStrong ad_strong;
  AdaptParamsBalance ad_balance;
  AdaptParamsConverge ad_converge;
  
  // backtracking parameters
  bool bt_enabled;
  real bt_beta, bt_gamma;
};

struct OptimizationProblem {
  OptimizationProblem()
      : mat(NULL), precond(NULL)
  {
  }
  
  int nrows, ncols;
  std::vector<Prox<real>*> prox_g;
  std::vector<Prox<real>*> prox_hc;
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
  void SetProx_g(const std::vector<Prox<real>*>& prox);
  void SetProx_hc(const std::vector<Prox<real>*>& prox);
  
  void SetOptions(const SolverOptions& opts);
  void SetCallback(SolverCallbackPtr cb); 

  bool Initialize();
  void Solve();
  void Release();

  void gpu_mem_amount(
      size_t& gpu_mem_required,
      size_t& gpu_mem_avail,
      size_t& gpu_mem_free);
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
