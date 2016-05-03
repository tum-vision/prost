/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#include "prost/backend/backend_admm.hpp"
#include "prost/linop/linearoperator.hpp"
#include "prost/prox/prox.hpp"
#include "prost/prox/prox_moreau.hpp"

#include "prost/cgls.hpp"
#include "prost/exception.hpp"
#include "prost/problem.hpp"

namespace prost {

// compute norm using cuBLAS
void nrm2(cublasHandle_t hdl, int n, const double *x, double *result) 
{
  cublasDnrm2(hdl, n, x, static_cast<int>(1), result);
}

void nrm2(cublasHandle_t hdl, int n, const float *x, double *result) 
{
  float result_float;
  cublasSnrm2(hdl, n, x, static_cast<int>(1), &result_float);
  *result = static_cast<double>(result_float);
}

/// \brief <0> = (alpha <1> + (1-alpha) <2> + <3>) / sqrt(<4>)
template<typename T>
struct temp1_functor
{
  temp1_functor(T alpha) : alpha_(alpha) { }
  
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = (alpha_ * thrust::get<1>(t) +
      (1 - alpha_) * thrust::get<2>(t) + thrust::get<3>(t)) / sqrt(thrust::get<4>(t));
  }

  T alpha_;
};

/// \brief <0> = sqrt(<3>) (<1> + <2>) 
template<typename T>
struct temp2_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = sqrt(thrust::get<3>(t)) * 
                        (thrust::get<1>(t) + thrust::get<2>(t));
  }
};

/// \brief Computes <0> = <1> - <2>
template<typename T>
struct difference_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) - thrust::get<2>(t);
  }
};

/// \brief Computes <0> = <0> * sqrt(<2>) + <1>
template<typename T>
struct x_proj_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = sqrt(thrust::get<2>(t)) * (thrust::get<0>(t) + thrust::get<1>(t));
  }
};

/// \brief Computes <0> = <1> * sqrt(<3>) - <2>
template<typename T>
struct x_dual_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) * sqrt(thrust::get<3>(t)) - thrust::get<2>(t);
  }
};

/// \brief Computes <0> = <1> / sqrt(<3>) - <2>
template<typename T>
struct z_dual_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) / sqrt(thrust::get<3>(t)) - thrust::get<2>(t);
  }
};

/// \brief Computes <0> = <0> + <1> - <2>
template<typename T>
struct sum_diff_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<0>(t) + thrust::get<1>(t) - thrust::get<2>(t);
  }
};

template<typename T>
struct gemv_functor1
{
  __host__ __device__
  T operator()(const T& v1, const T& v2) const
  {
    return sqrt(v1) * v2;
  }
};

template<typename T>
struct gemv_functor2
{
  gemv_functor2(T alpha, T beta) : alpha_(alpha), beta_(beta) { }

  __host__ __device__
  T operator()(const T& v1, const T& v2) const
  {
    return (beta_ / (alpha_ * sqrt(v1))) * v2;
  }

  T beta_;
  T alpha_;
};

template<typename T>
struct gemv_functor3
{
  gemv_functor3(T alpha) : alpha_(alpha) { }

  __host__ __device__
  T operator()(const T& v1, const T& v2) const
  {
    return alpha_ * sqrt(v1) * v2;
  }

  T alpha_;
};

// <0> = -rho(<1> - <2> + <3>)
template<typename T>
struct get_dual_functor
{
  get_dual_functor(T rho, T expo) : rho_(rho), expo_(expo) { }

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = -rho_ * pow(thrust::get<4>(t), expo_) * (thrust::get<1>(t) - thrust::get<2>(t) + thrust::get<3>(t));
  }

  T rho_;
  T expo_;
};

template<typename T>
struct GemvPrecondK 
{
  GemvPrecondK(const thrust::device_vector<T>& Sigma,
               const thrust::device_vector<T>& Tau,
               shared_ptr<LinearOperator<T>> linop,
               thrust::device_vector<T>& temp)
    : Sigma_(Sigma), Tau_(Tau), linop_(linop), temp_(temp)
  {
  }
  
  int operator()(
    char op, 
    const T alpha, 
    const thrust::device_vector<T>& x, 
    const T beta, 
    thrust::device_vector<T>& y) const
  {
    // implement y := alpha * op(A) x + beta * y

    if(op == 'n') // forward
    {      
      // temp = Tau^{1/2} * x
      thrust::transform(Tau_.begin(), Tau_.end(),
        x.begin(),
        temp_.begin(),
        gemv_functor1<T>());
      
      // y = (beta / (alpha * Sigma^{1/2})) * y
      thrust::transform(Sigma_.begin(), Sigma_.end(),
        y.begin(),
        y.begin(),
        gemv_functor2<T>(alpha, beta));

      // y += A * temp
      linop_->Eval(y, temp_, 1);

      // y = alpha * Sigma^{1/2} * y
      thrust::transform(Sigma_.begin(), Sigma_.end(),
        y.begin(),
        y.begin(),
        gemv_functor3<T>(alpha));
    }
    else // adjoint
    {
      // temp = Sigma^{1/2} * x
      thrust::transform(Sigma_.begin(), Sigma_.end(),
        x.begin(),
        temp_.begin(),
        gemv_functor1<T>());

      // y = (beta / (alpha * Tau^{1/2})) * y
      thrust::transform(Tau_.begin(), Tau_.end(),
        y.begin(),
        y.begin(),
        gemv_functor2<T>(alpha, beta));

      // y += A^T * temp
      linop_->EvalAdjoint(y, temp_, 1);

      // y = alpha * Tau^{1/2} * y
      thrust::transform(Tau_.begin(), Tau_.end(),
        y.begin(),
        y.begin(),
        gemv_functor3<T>(alpha));
    }

    return 0;
  }

  const thrust::device_vector<T>& Sigma_;
  const thrust::device_vector<T>& Tau_;
  thrust::device_vector<T>& temp_;
  shared_ptr<LinearOperator<T>> linop_;
};

template<typename T>
BackendADMM<T>::BackendADMM(const typename BackendADMM<T>::Options& opts)
  : opts_(opts)
{
}

template<typename T>
BackendADMM<T>::~BackendADMM()
{
}

template<typename T>
void BackendADMM<T>::Initialize()
{
  size_t m = this->problem_->nrows();
  size_t n = this->problem_->ncols();
  size_t l = std::max(m, n);

  // allocate variables
  try
  {
    x_half_.resize(n, 0);
    x_proj_.resize(n, 0);
    x_dual_.resize(n, 0);
    z_half_.resize(m, 0);
    z_proj_.resize(m, 0);
    z_dual_.resize(m, 0);
    
    temp1_.resize(n, 0);
    temp2_.resize(m, 0);
    temp3_.resize(l, 0);
  }
  catch(std::bad_alloc& e)
  {
    std::stringstream ss;
    ss << "Out of memory: " << e.what();
    throw Exception(ss.str());
  }

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

  if(this->problem_->prox_f().empty())
  {
    if(this->problem_->prox_fstar().empty())
      throw Exception("Neither prox_f nor prox_fstar specified.");

    for(auto& p : this->problem_->prox_fstar())
    {
      Prox<T> *moreau = new ProxMoreau<T>(p);
      moreau->Initialize(); // inner prox gets initializes twice. should be ok though.

      prox_f_.push_back( std::shared_ptr<Prox<T> >(moreau) );
    }
  }
  else
    prox_f_ = this->problem_->prox_f();

  delta_ = opts_.arb_delta;
  rho_ = opts_.rho0;
  iteration_ = 0;
  arb_u_ = arb_l_ = 0;

  cublasCreate_v2(&hdl_);
}

template<typename T>
void BackendADMM<T>::PerformIteration()
{
  // . temp1_ = T^{-1/2} (alpha x_half_ + (1-alpha) x_proj_ + x_dual_)
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.begin(),
          x_half_.begin(),
          x_proj_.begin(),
          x_dual_.begin(),
          this->problem_->scaling_right().begin())),
      
      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.end(),
          x_half_.end(),
          x_proj_.end(),
          x_dual_.end(),
          this->problem_->scaling_right().begin())),
      
      temp1_functor<T>(opts_.alpha));
  
  // . temp2_ = Sigma^{1/2} (z_half_ + z_dual_) 
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp2_.begin(),
          z_half_.begin(),
          z_dual_.begin(),
          this->problem_->scaling_left().begin())),
      
      thrust::make_zip_iterator(thrust::make_tuple(
          temp2_.end(),
          z_half_.end(),
          z_dual_.end(),
          this->problem_->scaling_left().end())),
      
      temp2_functor<T>());

  // abstract linear operator Sigma^{1/2} K Tau^{1/2}
  GemvPrecondK<T> gemv(
    this->problem_->scaling_left(),
    this->problem_->scaling_right(),
    this->problem_->linop(),
    temp3_);

  // z_dual_ is not needed, hence use it to store projection variable
  thrust::copy(temp2_.begin(), temp2_.end(), z_dual_.begin());
  thrust::device_vector<T>& tmp_proj_arg = z_dual_;

  // set x_proj_ to temp3_ for warm-starting
  thrust::copy(temp3_.begin(), temp3_.begin() + x_proj_.size(), x_proj_.begin());

  // tmp_proj_arg = temp2_ - Sigma^{1/2} K Tau^{1/2} temp_1
  gemv('n', -1, temp1_, 1, tmp_proj_arg);
  
  double cg_tol = opts_.cg_tol_min / 
    std::pow(static_cast<T>(iteration_ + 1), opts_.cg_tol_pow);
  cg_tol = std::max(cg_tol, opts_.cg_tol_max);

  //std::cout << iteration_ << ", " << cg_tol << ", " << opts_.cg_tol_min << ", " << opts_.cg_tol_max << std::endl;

  // x_half, z_half, z_proj and x_dual can be used as temporary variables
  // for the conjugate gradient method.
  thrust::device_vector<T>& tmp_p = x_half_;
  thrust::device_vector<T>& tmp_q = z_half_;
  thrust::device_vector<T>& tmp_r = z_proj_;
  thrust::device_vector<T>& tmp_s = x_dual_;

  // TODO: memset x_proj_ to zero or use warm-starting?!
  //thrust::fill(x_proj_.begin(), x_proj_.end(), 0);

  // Minimize |Kx-d|^2 + |x|^2 for d = temp2_ - K temp_1 with cgls Method
  int num_cg_iters_taken;
  cgls::Solve<T, GemvPrecondK<T> >(
    hdl_, 
    gemv, 
    this->problem_->nrows(),
    this->problem_->ncols(), 
    tmp_proj_arg, 
    x_proj_, 
    1, 
    cg_tol, 
    opts_.cg_max_iter, 
    true,
    tmp_p, 
    tmp_q, 
    tmp_r, 
    tmp_s,
    num_cg_iters_taken);

  // remember previous x_proj for warm-starting cg in the next iteration
  thrust::copy(x_proj_.begin(), x_proj_.end(), temp3_.begin());

  // x_proj = Tau^{1/2} (x_proj + temp1_)
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          x_proj_.begin(),
          temp1_.begin(),
          this->problem_->scaling_right().begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          x_proj_.end(),
          temp1_.end(),
          this->problem_->scaling_right().end())),

      x_proj_functor<T>());

  // z_proj = K x_proj
  this->problem_->linop()->Eval(z_proj_, x_proj_);
  
  // x_dual_ = temp1_ * Tau^{1/2} - x_proj_
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          x_dual_.begin(),
          temp1_.begin(),
          x_proj_.begin(),
          this->problem_->scaling_right().begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          x_dual_.end(),
          temp1_.end(),
          x_proj_.end(),
          this->problem_->scaling_right().begin())),

      x_dual_functor<T>());

  // z_dual_ = temp2_ / Sigma^{1/2} - z_proj_
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          z_dual_.begin(),
          temp2_.begin(),
          z_proj_.begin(),
          this->problem_->scaling_left().begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          z_dual_.end(),
          temp2_.end(),
          z_proj_.end(),
          this->problem_->scaling_left().end())),

      z_dual_functor<T>());

  // temp1_ = x_proj_ - x_dual_
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.begin(),
          x_proj_.begin(),
          x_dual_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.end(),
          x_proj_.end(),
          x_dual_.end())),

      difference_functor<T>());

  // x_half_ = prox_g(temp_)
  for(auto& p : prox_g_)
    p->Eval(x_half_, temp1_, this->problem_->scaling_right(), 1 / rho_);

  // temp2_ = z_proj_ - z_dual_
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp2_.begin(),
          z_proj_.begin(),
          z_dual_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          temp2_.end(),
          z_proj_.end(),
          z_dual_.end())),

      difference_functor<T>());

  // z_half_ = prox_f(temp_)
  for(auto& p : prox_f_)
    p->Eval(z_half_, temp2_, this->problem_->scaling_left(), rho_, true); 

  iteration_++;

  // compute residuals every "opts_.residual_iter" iterations and
  // adapt stepsizes for residual base adaptive schemes
  if(iteration_ == 0 || (iteration_ % opts_.residual_iter) == 0)
  {   
    double primal_residual;
    double primal_var_norm;
    double dual_residual;
    double dual_var_norm;

    thrust::copy(z_half_.begin(), z_half_.end(), temp2_.begin());
    this->problem_->linop()->Eval(temp2_, x_half_, -1);

    // scale with Sigma^{1/2}
    thrust::transform(
      this->problem_->scaling_left().begin(), 
      this->problem_->scaling_left().end(), 
      temp2_.begin(), 
      temp2_.begin(),
      gemv_functor1<T>());

    nrm2(hdl_, this->problem_->nrows(), thrust::raw_pointer_cast(temp2_.data()), &primal_residual);

    // scale with Sigma^{1/2}
    thrust::transform(
      this->problem_->scaling_left().begin(), 
      this->problem_->scaling_left().end(), 
      z_half_.begin(), 
      temp2_.begin(),
      gemv_functor1<T>());

    nrm2(hdl_, this->problem_->nrows(), thrust::raw_pointer_cast(temp2_.data()), &primal_var_norm);

    // Compute dual variable temp1_ = w
    thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.begin(),
          x_half_.begin(),
          x_proj_.begin(),
          x_dual_.begin(),
          this->problem_->scaling_right().begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.end(),
          x_half_.end(),
          x_proj_.end(),
          x_dual_.end(),
          this->problem_->scaling_right().end())),

      get_dual_functor<T>(rho_, -1));

    // scale with Tau^{1/2}
    thrust::transform(
      this->problem_->scaling_right().begin(), 
      this->problem_->scaling_right().end(), 
      temp1_.begin(), 
      temp2_.begin(),
      gemv_functor1<T>());

    // reduce temp2_ to get norm w
    nrm2(hdl_, this->problem_->ncols(), thrust::raw_pointer_cast(temp2_.data()), &dual_var_norm);

    // compute dual variable temp2_ = y
    thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp2_.begin(),
          z_half_.begin(),
          z_proj_.begin(),
          z_dual_.begin(),
          this->problem_->scaling_left().begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          temp2_.end(),
          z_half_.end(),
          z_proj_.end(),
          z_dual_.end(), 
          this->problem_->scaling_left().end())),

      get_dual_functor<T>(rho_, 1));

    // Compute w + K^T y
    this->problem_->linop()->EvalAdjoint(temp1_, temp2_, 1);

    // scale with Tau^{1/2}
    thrust::transform(
      this->problem_->scaling_right().begin(), 
      this->problem_->scaling_right().end(), 
      temp1_.begin(), 
      temp1_.begin(),
      gemv_functor1<T>());

    // reduce temp1_ to get residual
    nrm2(hdl_, this->problem_->ncols(), thrust::raw_pointer_cast(temp1_.data()), &dual_residual);

    // fill variables, adapt stepsizes, rescale tilde variables
    this->primal_residual_ = primal_residual;
    this->primal_var_norm_ = primal_var_norm;
    this->dual_residual_ = dual_residual;
    this->dual_var_norm_ = dual_var_norm;

    T eps_primal = this->eps_primal();
    T eps_dual = this->eps_dual();

    T rho_prev = rho_;
    if( (this->dual_residual_ < eps_dual) && (opts_.arb_tau * iteration_ > arb_l_) )
    {
      rho_ *= delta_;
      delta_ *= opts_.arb_gamma;
      arb_u_ = iteration_;
    }
    else if( (this->primal_residual_ < eps_primal) && (opts_.arb_tau * iteration_ > arb_u_) )
    {
      rho_ /= delta_;
      delta_ *= opts_.arb_gamma;
      arb_l_ = iteration_;
    }

    // rescale dual variables
    if(std::abs(rho_ - rho_prev) > 1e-7)
    {
      thrust::transform(
          x_dual_.begin(),
          x_dual_.end(),
          x_dual_.begin(),
          (rho_prev / rho_) * thrust::placeholders::_1);

      thrust::transform(
          z_dual_.begin(),
          z_dual_.end(),
          z_dual_.begin(),
          (rho_prev / rho_) * thrust::placeholders::_1);
    }
  }
}

template<typename T>
void BackendADMM<T>::Release()
{
  cublasDestroy_v2(hdl_);
}

template<typename T>
void BackendADMM<T>::current_solution(
  vector<T>& primal, vector<T>& dual)
{
  thrust::copy(x_half_.begin(), x_half_.end(), primal.begin());

  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(
        temp2_.begin(),
        z_half_.begin(),
        z_proj_.begin(),
        z_dual_.begin(),
        this->problem_->scaling_left().begin())),

    thrust::make_zip_iterator(thrust::make_tuple(
        temp2_.end(),
        z_half_.end(),
        z_proj_.end(),
        z_dual_.end(), 
        this->problem_->scaling_left().end())),
    
    get_dual_functor<T>(rho_, 1));

  thrust::copy(temp2_.begin(), temp2_.end(), dual.begin());
}

template<typename T>
void BackendADMM<T>::current_solution(
  vector<T>& primal_x,
  vector<T>& primal_z,
  vector<T>& dual_y,
  vector<T>& dual_w)
{
  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(
        temp1_.begin(),
        x_half_.begin(),
        x_proj_.begin(),
        x_dual_.begin(),
        this->problem_->scaling_right().begin())),

    thrust::make_zip_iterator(thrust::make_tuple(
        temp1_.end(),
        x_half_.end(),
        x_proj_.end(),
        x_dual_.end(),
        this->problem_->scaling_right().end())),

    get_dual_functor<T>(rho_, -1));

  thrust::copy(temp1_.begin(), temp1_.end(), dual_w.begin());

  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(
        temp2_.begin(),
        z_half_.begin(),
        z_proj_.begin(),
        z_dual_.begin(),
        this->problem_->scaling_left().begin())),

    thrust::make_zip_iterator(thrust::make_tuple(
        temp2_.end(),
        z_half_.end(),
        z_proj_.end(),
        z_dual_.end(), 
        this->problem_->scaling_left().end())),

    get_dual_functor<T>(rho_, 1));

  thrust::copy(temp2_.begin(), temp2_.end(), dual_y.begin());
  thrust::copy(x_half_.begin(), x_half_.end(), primal_x.begin());
  thrust::copy(z_half_.begin(), z_half_.end(), primal_z.begin());
}

template<typename T>
size_t BackendADMM<T>::gpu_mem_amount() const
{
  size_t m = this->problem_->nrows();
  size_t n = this->problem_->ncols();

  return (4 * (n + m) + std::max(m, n)) * sizeof(T);
}

// Explicit template instantiation
template class BackendADMM<float>;
template class BackendADMM<double>;

} // namespace prost