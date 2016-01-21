#ifndef BACKEND_PDHG_HPP_
#define BACKEND_PDHG_HPP_

#include <thrust/device_vector.h>

struct BackendPDHGOptions {
  double tau0, sigma0; // initial step sizes
  bool solve_dual_problem; // overrelaxation on dual variables?
};

// callback function for updating adaptive stepsizes, has the following signature
// (iter, res_primal, res_dual, tau, sigma)
typedef std::function<void(int, double, double, double&, double&)> BackendPDHGStepsizeCallback;

template<typename T> 
class BackendPDHG : public Backend<T> {
public:
  BackendPDHG(std::unique_ptr<Problem<T> > problem);
  virtual ~BackendPDHG();

  virtual void Initialize();
  virtual void PerformIteration();
  virtual void Release();

  virtual void current_solution(std::vector<T>& primal, std::vector<T>& dual) const;
  virtual T primal_residual() const;
  virtual T dual_residual() const;

  // returns amount of gpu memory required in bytes
  virtual size_t gpu_mem_amount() const;

protected:
  thrust::device_vector<T> x_; // primal variable x^k
  thrust::device_vector<T> y_; // dual variables y^k
  thrust::device_vector<T> x_prev_; // previous primal variable x^{k-1}
  thrust::device_vector<T> y_prev_; // previous dual variable y^{k-1}
  thrust::device_vector<T> temp_; // temporary variable to store result of proxs and residuals
  thrust::device_vector<T> kx_; //  holds mat-vec product K x^k
  thrust::device_vector<T> kty_; //  holds mat-vec product K^T y^k
  thrust::device_vector<T> kx_prev_; //  holds mat-vec product K x^{k-1}
  thrust::device_vector<T> kty_prev; //  holds mat-vec product K^T y^{k-1}

  // algorithm parameters
  T tau_;
  T sigma_;
  T theta_;
  BackendPDHGOptions opts_;
  BackendPDHGStepsizeCallback stepsize_cb_;
};

#endif
