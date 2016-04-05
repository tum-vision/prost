#ifndef PROST_BACKEND_ADMM_HPP_
#define PROST_BACKEND_ADMM_HPP_

#include <thrust/device_vector.h>

#include "prost/backend/backend.hpp"
#include "prost/common.hpp"

namespace prost {

template<typename T> class Prox;

template<typename T>
class BackendADMM : public Backend<T>
{
public:
  struct Options
  {
    /// \brief initial step size
    double rho0;

    // \brief over-relaxation factor
    double alpha;

    /// \brief factor for increasing cg tolerance every iteration
    double cg_tol_pow;

    /// \brief initial cg tolerance
    double cg_tol_ini;

    /// \brief maximal cg tolerance
    double cg_tol_max;

    /// \brief maximum number of cg iterations
    int cg_max_iter;

    /// \brief Every how many iterations to compute the residuals?
    int residual_iter;

    /// \brief Parameters for residual converging step size scheme.
    T arb_delta, arb_tau, arb_gamma;
  };

  BackendADMM(const typename BackendADMM<T>::Options& opts);
  virtual ~BackendADMM();

  virtual void Initialize();
  virtual void PerformIteration();
  virtual void Release();

  virtual void current_solution(vector<T>& primal, vector<T>& dual);

  virtual void current_solution(vector<T>& primal_x,
                                vector<T>& primal_z,
                                vector<T>& dual_y,
                                vector<T>& dual_w);

  /// \brief Returns amount of gpu memory required in bytes.
  virtual size_t gpu_mem_amount() const;

private:
  thrust::device_vector<T> x_half_, z_half_;
  thrust::device_vector<T> x_proj_, z_proj_;
  thrust::device_vector<T> x_dual_, z_dual_;
  thrust::device_vector<T> temp1_, temp2_;
  
  /// \brief ADMM-specific options.
  typename BackendADMM<T>::Options opts_;

  // Current step size
  T rho_;

  /// \brief Internal iteration counter.
  size_t iteration_;

  /// \brief Internal prox_g
  vector< shared_ptr<Prox<T> > > prox_g_;

  /// \brief Internal prox_f
  vector< shared_ptr<Prox<T> > > prox_f_;
};

} // namespace prost

#endif // PROST_BACKEND_ADMM_HPP
