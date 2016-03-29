#include <algorithm>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#include "prost/backend/backend_pdhg.hpp"
#include "prost/linop/linearoperator.hpp"
#include "prost/prox/prox.hpp"
#include "prost/prox/prox_moreau.hpp"
#include "prost/exception.hpp"
#include "prost/problem.hpp"

namespace prost {

/// \brief Computes x^{k+1} = x^k - tau T K^T y^k
template<typename T>
struct primal_proxarg_functor
{
  __host__ __device__ primal_proxarg_functor(T tau) : tau_(tau) { }

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<3>(t) = thrust::get<0>(t) - tau_ * thrust::get<1>(t) * thrust::get<2>(t);
  }

  T tau_;
};

/// \brief Computes y^{k+1} = y^k + sigma S K (x^{k+1} + theta (x^{k+1} - x^k))
template<typename T>
struct dual_proxarg_functor
{
  __host__ __device__ dual_proxarg_functor(T sigma, T theta) 
      : sigma_(sigma), theta_(theta) { }

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<4>(t) = thrust::get<0>(t) + sigma_ * thrust::get<1>(t) * 
      ((1 + theta_) * thrust::get<2>(t) - theta_ * thrust::get<3>(t));
  }

  T sigma_;
  T theta_;
};

/// \brief Computes the (scaled) dual residual 
template<typename T>
struct dual_residual_transform : public thrust::unary_function<thrust::tuple<T,T,T,T,T>, thrust::tuple<T,T> >
{
  typedef typename thrust::tuple<T,T,T,T,T> InputTuple;
  typedef typename thrust::tuple<T,T> OutputTuple;

  __host__ __device__ dual_residual_transform(T tau)
  : tau_(tau) { }

  __host__ __device__
  OutputTuple operator()(const InputTuple& t)
  {
    const T tau_diag = thrust::get<2>(t);
    const T w_hat = (thrust::get<0>(t) - thrust::get<1>(t)) / (tau_ * sqrt(tau_diag)) - 
                    sqrt(tau_diag) * thrust::get<3>(t);
    const T diff = w_hat + sqrt(tau_diag) * thrust::get<4>(t); // w_hat^{k+1} + T K^T y^{k+1}

    return OutputTuple(diff * diff, w_hat * w_hat);
  }  

  T tau_;
};

/// \brief Computes the (scaled) primal residual
template<typename T>
struct primal_residual_transform : public thrust::unary_function<thrust::tuple<T,T,T,T,T>, thrust::tuple<T, T> >
{
  typedef typename thrust::tuple<T,T,T,T,T> InputTuple;
  typedef typename thrust::tuple<T,T> OutputTuple;

  __host__ __device__ primal_residual_transform(T sigma, T theta)
      : sigma_(sigma), theta_(theta) { }

  __host__ __device__
  OutputTuple operator()(const InputTuple& t)
  {
    const T sigma_diag = thrust::get<2>(t);
    const T z_hat = (thrust::get<0>(t) - thrust::get<1>(t)) / (sigma_ * sqrt(sigma_diag)) +
                    sqrt(sigma_diag) * ((1 + theta_) * thrust::get<4>(t) - theta_ * thrust::get<3>(t));
      
    const T diff = z_hat - sqrt(sigma_diag) * thrust::get<4>(t);

    return OutputTuple(diff * diff, z_hat * z_hat);
  }  

  T sigma_;
  T theta_;
};

/// \brief Helper 
template<typename T>
struct residual_tuple_sum : public thrust::binary_function< thrust::tuple<T, T>, 
                                                              thrust::tuple<T, T>,
  thrust::tuple<T, T> >
{
  typedef typename thrust::tuple<T, T> Tuple;

  __host__ __device__
  Tuple operator()(const Tuple& t0, const Tuple& t1) const
  {
    return Tuple(
        thrust::get<0>(t0) + thrust::get<0>(t1), 
        thrust::get<1>(t0) + thrust::get<1>(t1));
  }
};

/*
          x_prev_.end(), 0
          x_.end(), 1
          this->problem_->scaling_right().end(), 2
          kty_prev_.end(), 3
          temp_.end() 4
 */
template<typename T>
struct compute_w_variable_functor
{
  __host__ __device__ compute_w_variable_functor(T tau) : tau_(tau) { }

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<4>(t) = (thrust::get<0>(t) - thrust::get<1>(t)) / (thrust::get<2>(t) * tau_)
                        - thrust::get<3>(t);
  }

  T tau_;
};

/*
          y_prev_.end(), 0
          y_.end(), 1
          this->problem_->scaling_left().end(), 2
          kx_.end(), 3
          kx_prev_.end(), 4
          temp_.end())), 5
*/
template<typename T>
struct compute_z_variable_functor
{
  __host__ __device__ compute_z_variable_functor(T sigma, T theta)
      : sigma_(sigma), theta_(theta) { }

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<5>(t) = (thrust::get<0>(t) - thrust::get<1>(t)) / (sigma_ * thrust::get<2>(t))
                        + (1 + theta_) * thrust::get<3>(t) - theta_ * thrust::get<4>(t);
  }

  T sigma_;
  T theta_;
};

template<typename T>
BackendPDHG<T>::BackendPDHG(const typename BackendPDHG<T>::Options& opts)
    : opts_(opts)
{
}

template<typename T>
BackendPDHG<T>::~BackendPDHG()
{
}

template<typename T>
void 
BackendPDHG<T>::Initialize()
{
  size_t m = this->problem_->nrows();
  size_t n = this->problem_->ncols();
  size_t l = std::max(m, n);

  // allocate variables
  try
  {
    x_.resize(n, 0);
    x_prev_.resize(n, 0);
    kty_prev_.resize(n, 0);
    kty_.resize(n, 0);
    y_.resize(m, 0);
    y_prev_.resize(m, 0);
    kx_.resize(m, 0);
    kx_prev_.resize(m, 0);
    temp_.resize(l, 0);
  }
  catch(std::bad_alloc& e)
  {
    std::stringstream ss;
    ss << "Out of memory: " << e.what();
    throw Exception(ss.str());
  }

  iteration_ = 0;
  tau_ = opts_.tau0;
  sigma_ = opts_.sigma0;
  theta_ = 1;

  arb_l_ = arb_u_ = 0;
  arg_alpha_ = opts_.arg_alpha0;

  // check if proxs are available (or create via moreau)
  if(this->problem_->prox_g().empty())
  {
    if(this->problem_->prox_gstar().empty())
      throw Exception("Neither prox_g nor prox_gstar specified.");

    for(auto& p : this->problem_->prox_gstar())
    {
      Prox<T> *moreau = new ProxMoreau<T>(p);
      moreau->Initialize(); // inner prox gets initializes twice. should be ok though.

      prox_g_.push_back( std::shared_ptr<Prox<T> >(moreau) );
    }
  }
  else
    prox_g_ = this->problem_->prox_g();

  if(this->problem_->prox_fstar().empty())
  {
    if(this->problem_->prox_f().empty())
      throw Exception("Neither prox_f nor prox_fstar specified.");

    for(auto& p : this->problem_->prox_f())
    {
      Prox<T> *moreau = new ProxMoreau<T>(p);
      moreau->Initialize(); // inner prox gets initializes twice. should be ok though.

      prox_fstar_.push_back( std::shared_ptr<Prox<T> >(moreau) );
    }
  }
  else
    prox_fstar_ = this->problem_->prox_fstar();

  // set residuals to zero
  this->primal_var_norm_ = 0;
  this->dual_var_norm_ = 0;
  this->primal_residual_ = 0;
  this->dual_residual_ = 0;

  if(opts_.scale_steps_operator)
  {
    T norm = this->problem_->normest();

    if(std::abs(norm - 1) > 0.1)
    {
      tau_ /= norm;
      sigma_ /= norm;

      if(this->solver_opts_.verbose)
        cout << "|K|=" << norm << " => Rescaled tau=" << tau_ << ", sigma=" << sigma_ << "." << endl; 
    }
  }

  if(this->solver_opts_.x0.size() > 0)
  {
    if(this->solver_opts_.x0.size() == n)
    {
      x_ = this->solver_opts_.x0;
      x_prev_ = this->solver_opts_.x0;
    }
    else
      throw Exception("Initial primal solution has wrong size.");
  }

  if(this->solver_opts_.y0.size() > 0)
  {
    if(this->solver_opts_.y0.size() == m)
    {
      y_ = this->solver_opts_.y0;
      y_prev_ = this->solver_opts_.y0;
    }
    else
      throw Exception("Initial dual solution has wrong size.");
  }
}

template<typename T>
void 
BackendPDHG<T>::PerformIteration()
{
  // compute primal prox arg into temp_
  // thrust::get<3>(t) = thrust::get<0>(t) - tau_ * thrust::get<1>(t) * thrust::get<2>(t);
  thrust::for_each(

      thrust::make_zip_iterator(thrust::make_tuple(
          x_.begin(), 
          this->problem_->scaling_right().begin(), 
          kty_.begin(), 
          temp_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          x_.end(), 
          this->problem_->scaling_right().end(), 
          kty_.end(), 
          temp_.end())),

      primal_proxarg_functor<T>(tau_));

  // remember previous primal iterate
  x_.swap(x_prev_);

  // apply prox_g
  for(auto& p : prox_g_)
    p->Eval(x_, temp_, this->problem_->scaling_right(), tau_);

  // remember Kx^k
  kx_.swap(kx_prev_);

  // compute Kx^{k+1}
  this->problem_->linop()->Eval(kx_, x_);

  // compute dual prox arg
  // thrust::get<4>(t) = thrust::get<0>(t) + sigma_ * thrust::get<1>(t) * 
  //                     ((1 + theta_) * thrust::get<2>(t) - theta_ * thrust::get<3>(t));
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          y_.begin(),
          this->problem_->scaling_left().begin(),
          kx_.begin(),
          kx_prev_.begin(),
          temp_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          y_.end(),
          this->problem_->scaling_left().end(),
          kx_.end(),
          kx_prev_.end(),
          temp_.end())),

      dual_proxarg_functor<T>(sigma_, theta_));

  y_.swap(y_prev_);
    
  // apply prox_fstar
  for(auto& p : prox_fstar_)
    p->Eval(y_, temp_, this->problem_->scaling_left(), sigma_);

  UpdateResidualsAndStepsizes();

  iteration_++;

  // remember K^T y^k
  kty_.swap(kty_prev_);

  // compute K^T y^{k+1}
  this->problem_->linop()->EvalAdjoint(kty_, y_);
}

template<typename T>
void
BackendPDHG<T>::UpdateResidualsAndStepsizes()
{
  // compute residuals every "opts_.residual_iter" iterations and
  // adapt stepsizes for residual base adaptive schemes
  if(iteration_ == 0 || (iteration_ % opts_.residual_iter) == 0)
  {
    // compute primal residual |Kx - z|^2 and norm |z|^2
    thrust::tuple<T, T> primal = thrust::transform_reduce(

        thrust::make_zip_iterator(thrust::make_tuple(
            y_prev_.begin(),
            y_.begin(),
            this->problem_->scaling_left().begin(),
            kx_prev_.begin(),
            kx_.begin())),

        thrust::make_zip_iterator(thrust::make_tuple(
            y_prev_.end(),
            y_.end(),
            this->problem_->scaling_left().end(),
            kx_prev_.end(),
            kx_.end())),

        primal_residual_transform<T>(sigma_, theta_),
        thrust::tuple<T, T>(0, 0),
        residual_tuple_sum<T>());

    // compute dual residual |K^T y + w|^2 and norm |w|^2
    thrust::tuple<T, T> dual = thrust::transform_reduce(

        thrust::make_zip_iterator(thrust::make_tuple(
            x_prev_.begin(),
            x_.begin(),
            this->problem_->scaling_right().begin(),
            kty_prev_.begin(),
            kty_.begin())),

        thrust::make_zip_iterator(thrust::make_tuple(
            x_prev_.end(),
            x_.end(),
            this->problem_->scaling_right().end(),
            kty_prev_.end(),
            kty_.end())),

        dual_residual_transform<T>(tau_),
        thrust::tuple<T, T>(0, 0),
        residual_tuple_sum<T>());

    this->primal_residual_ = std::sqrt(thrust::get<0>(primal));
    this->primal_var_norm_ = std::sqrt(thrust::get<1>(primal));
    this->dual_residual_ = std::sqrt(thrust::get<0>(dual));
    this->dual_var_norm_ = std::sqrt(thrust::get<1>(dual));

    T eps_primal = this->eps_primal();
    T eps_dual = this->eps_dual();

    switch(opts_.stepsize_variant)
    {      
    case BackendPDHG<T>::StepsizeVariant::kPDHGStepsResidualGoldstein: {
      T scale = eps_dual / eps_primal;
	
      if( this->dual_residual_ > (scale * this->primal_residual_ * opts_.arg_delta) )
      {
        tau_ = tau_ / (1 - arg_alpha_);
        sigma_ = sigma_ * (1 - arg_alpha_);
        arg_alpha_ = arg_alpha_ * opts_.arg_nu;
      }

      if( this->dual_residual_ < (scale * this->primal_residual_ / opts_.arg_delta) )
      {
        tau_ = tau_ * (1 - arg_alpha_);
        sigma_ = sigma_ / (1 - arg_alpha_);
        arg_alpha_ = arg_alpha_ * opts_.arg_nu;
      }
      
    } break;

      case BackendPDHG<T>::StepsizeVariant::kPDHGStepsResidualBoyd:
        if( (this->dual_residual_ < eps_dual) && (opts_.arb_tau * iteration_ > arb_l_) )
        {
          tau_ /= opts_.arb_delta;
          sigma_ *= opts_.arb_delta;
          arb_u_ = iteration_;
        }
        else if( (this->primal_residual_ < eps_primal) && (opts_.arb_tau * iteration_ > arb_u_) )
        {
          tau_ *= opts_.arb_delta;
          sigma_ /= opts_.arb_delta;
          arb_l_ = iteration_;
        }

        break;

      default:
        break;
    }
  }

  if(opts_.stepsize_variant == BackendPDHG<T>::StepsizeVariant::kPDHGStepsAlg2)
  {
    if(this->solver_opts_.solve_dual_problem)
    {
      theta_ = 1. / std::sqrt(1. + 2. * opts_.alg2_gamma * sigma_);
      tau_ = theta_ / tau_;
      sigma_ = sigma_ * theta_;
    }
    else
    {
      theta_ = 1. / std::sqrt(1. + 2. * opts_.alg2_gamma * tau_);
      tau_ = theta_ * tau_;
      sigma_ = sigma_ / theta_;
    }

    //cout << "theta=" << theta_ << ", tau=" << tau_ << ", sigma=" << sigma_ << ", gamma=" << opts_.alg2_gamma << endl;
  }
}

template<typename T>
void 
BackendPDHG<T>::Release() { }

template<typename T>
void 
BackendPDHG<T>::current_solution(std::vector<T>& primal, std::vector<T>& dual) 
{
  thrust::copy(x_.begin(), x_.end(), primal.begin());
  thrust::copy(y_.begin(), y_.end(), dual.begin());
}

template<typename T>
size_t 
BackendPDHG<T>::gpu_mem_amount() const
{
  size_t m = this->problem_->nrows();
  size_t n = this->problem_->ncols();

  return (4 * (n + m) + std::max(n, m)) * sizeof(T);
}

template<typename T>
void
BackendPDHG<T>::current_solution(
    vector<T>& primal_x,
    vector<T>& primal_z,
    vector<T>& dual_y,
    vector<T>& dual_w) 
{
  thrust::copy(x_.begin(), x_.end(), primal_x.begin());
  thrust::copy(y_.begin(), y_.end(), dual_y.begin());

  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          x_prev_.begin(),
          x_.begin(),
          this->problem_->scaling_right().begin(),
          kty_prev_.begin(),
          temp_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          x_prev_.end(),
          x_.end(),
          this->problem_->scaling_right().end(),
          kty_prev_.end(),
          temp_.end())),

      compute_w_variable_functor<T>(tau_));

  thrust::copy(temp_.begin(), temp_.begin() + dual_w.size(), dual_w.begin());
  
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          y_prev_.begin(),
          y_.begin(),
          this->problem_->scaling_left().begin(),
          kx_.begin(),
          kx_prev_.begin(),
          temp_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          y_prev_.end(),
          y_.end(),
          this->problem_->scaling_left().end(),
          kx_.end(),
          kx_prev_.end(),
          temp_.end())),

      compute_z_variable_functor<T>(sigma_, theta_));

  thrust::copy(temp_.begin(), temp_.begin() + primal_z.size(), primal_z.begin());  
}

// Explicit template instantiation
template class BackendPDHG<float>;
template class BackendPDHG<double>;

} // namespace prost