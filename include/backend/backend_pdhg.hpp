#ifndef BACKEND_PDHG_HPP_
#define BACKEND_PDHG_HPP_

#include <thrust/device_vector.h>

///
/// \brief Implementation of the primal-dual hybrid-gradient method.
/// 
template<typename T> 
class BackendPDHG : public Backend<T> 
{
public:
  /// \brief Step size scheme for the PDHG.
  enum StepsizeVariant 
  {
    /// \brief Constant steps.
    kPDHGStepsAlg1,

    /// \brief Variable steps for strongly convex problems.
    kPDHGStepsAlg2,

    /// \brief Residual balancing scheme from "Goldstein, Esser" paper.
    kPDHGStepsResidualBalance,

    /// \brief Residual converging scheme from "Fougner, Boyd" paper
    kPDHGStepsResidualConverge,

    /// \brief User-defined step size scheme via callback function.
    kPDHGStepsCallback,
  };

  /// \brief Callback function for updating adaptive stepsizes, it has the following signature
  ///        (iter, res_primal, res_dual, new_tau, new_sigma).
  typedef std::function<void(int, double, double, double&, double&)> StepsizeCallback;

  /// \brief Detailed options for the primal-dual algorithm.
  struct Options 
  {
    /// \brief Initial primal step size.
    double tau0;
  
    /// \brief Initial dual step size.
    double sigma0; 

    /// \brief Perform overrelaxation on dual variables?
    bool solve_dual_problem; 

    /// \brief Every how many iterations to compute the residuals?
    int residual_iter;

    /// \brief Scale step sizes to ensure tau*sigma*||K||^2 = 1. 
    bool scale_steps_operator;

    /// \brief Strong convexity parameter for algorithm 2 step size scheme.
    T alg2_gamma;

    /// \brief Parameters for residual balancing step size scheme.
    T arb_alpha0, arb_nu, arb_delta;

    /// \brief Parameters for residual converging step size scheme.
    T arc_delta, arc_tau;
  };

  BackendPDHG(std::shared_ptr<Problem<T> > problem);
  virtual ~BackendPDHG();

  void SetOptions(typename const BackendPDHG<T>::Options& opts);
  void SetStepsizeCallback(typename const BackendPDHG<T>::StepsizeCallback& cb);

  virtual void Initialize();
  virtual void PerformIteration();
  virtual void Release();

  virtual void current_solution(std::vector<T>& primal, std::vector<T>& dual) const;

  virtual T primal_var_norm() const;
  virtual T dual_var_norm() const;
  virtual T primal_residual() const;
  virtual T dual_residual() const;

  /// \brief Returns amount of gpu memory required in bytes.
  virtual size_t gpu_mem_amount() const;

protected:
  // \brief Primal variable x^k.
  thrust::device_vector<T> x_; 

  // \brief Dual variables y^k.
  thrust::device_vector<T> y_; 

  // previous primal variable x^{k-1}
  thrust::device_vector<T> x_prev_; 

  // previous dual variable y^{k-1}
  thrust::device_vector<T> y_prev_; 

  // temporary variable to store result of proxs and residuals
  thrust::device_vector<T> temp_;

  // holds mat-vec product K x^k
  thrust::device_vector<T> kx_;

  // holds mat-vec product K^T y^k
  thrust::device_vector<T> kty_;

  // holds mat-vec product K x^{k-1}
  thrust::device_vector<T> kx_prev_;

  // holds mat-vec product K^T y^{k-1}
  thrust::device_vector<T> kty_prev; 

  /// \brief Current primal step size.
  T tau_;

  /// \brief Current dual step size.
  T sigma_;

  /// \brief Current overrelaxation parameter.
  T theta_;

  /// \brief PDHG-specific options.
  typename BackendPDHG<T>::Options opts_;

  /// \brief PDHG stepszie callback
  typename BackendPDHG<T>::StepsizeCallback stepsize_cb_;
};

#endif
