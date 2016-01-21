/**
#ifndef SOLVER_BACKEND_PDHG_HPP_
#define SOLVER_BACKEND_PDHG_HPP_

#include "solver_backend.hpp"

/**
 * @brief ...
 */
class SolverBackendPDHG : public SolverBackend {
public:
  SolverBackendPDHG(
      const OptimizationProblem& problem,
      const SolverOptions& opts)
      : SolverBackend(problem, opts) { }
  virtual ~SolverBackendPDHG() { }
  
  virtual bool Initialize();
  virtual void PerformIteration();
  virtual void Release();

  virtual void iterates(real *primal, real *dual);
  virtual bool converged();
  virtual std::string status();

  // returns amount of gpu memory required in bytes
  virtual size_t gpu_mem_amount();
  
protected:
  // cublas is required to sum residuals
  cublasHandle_t cublas_handle_;
  
  // algorithm variables
  real *d_x_; // primal variable x^k
  real *d_y_; // dual variable y^k
  real *d_x_prev_; // previous primal variable x^{k-1}
  real *d_y_prev_; // previous dual variable y^{k-1}
  real *d_temp_; // temporary variable to store result of proxs and residuals
  real *d_kx_; // holds mat-vec product K x^k
  real *d_kty_; // holds mat-vec product K^T y^k
  real *d_kx_prev_; // holds mat-vec product K x^{k-1}
  real *d_kty_prev_; // holds mat-vec product K^T y^{k-1}
  
  real tau_; // primal step size
  real sigma_; // dual step size
  real theta_; // overrelaxation parameter
  real alpha_; // adaptive step size parameter
  real res_primal_; // summed primal residual
  real res_dual_; // summed dual residual
  real gamma_;

  // for adaptive step size rule from Boyd's paper
  int adc_l_; 
  int adc_u_;
};

#endif
*/