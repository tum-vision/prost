#include <algorithm>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#include "prost/backend/backend_admm.hpp"
#include "prost/linop/linearoperator.hpp"
#include "prost/prox/prox.hpp"
#include "prost/prox/prox_moreau.hpp"

#include "prost/exception.hpp"
#include "prost/problem.hpp"

namespace prost {

/// \brief <0> = alpha <1> + (1-alpha) <2> + <3>
template<typename T>
struct temp1_functor
{
  temp1_functor(T alpha) : alpha_(alpha) { }
  
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = alpha_ * thrust::get<1>(t) +
                        (1 - alpha_) * thrust::get<2>(t) + thrust::get<3>(t);
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
    thrust::get<0>(t) = thrust::get<0>(t) * sqrt(thrust::get<2>(t)) + thrust::get<1>(t);
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

template<typename T>
struct GemvPrecondK //: cgls::Gemv<T> {
{
  GemvPrecondK(const thrust::device_vector<T>& Sigma,
               const thrust::device_vector<T>& Tau,
               shared_ptr<LinearOperator<T>> linop,
               const thrust::device_vector<T>& temp)
      : Sigma_(Sigma), Tau_(Tau), temp_(temp)
  {
  }
  
  int operator()(char op, const T alpha, const T *x, const T beta, T *y) const
  {
    // implement y := alpha * op(A) x + beta * y

    if(op == 'n') // forward
    {      
      // temp = Tau^{1/2} * x
      thrust::transform(Tau_.begin(), Tau_.end(),
                        thrust::device_pointer_cast(x),
                        temp_.begin(),
                        gemv_functor1<T>());
      
      // y = (beta / (alpha * Sigma^{1/2})) * y
      thrust::transform(Sigma_.begin(), Sigma_.end(),
                        thrust::device_pointer_cast(y),
                        thrust::device_pointer_cast(y),
                        gemv_functor2<T>(alpha, beta));

      // y += A * temp
      linop_->Eval(y, temp_, 1);

      // y = alpha * Sigma^{1/2} * y
      thrust::transform(Sigma_.begin(), Sigma_.end(),
                        thrust::device_pointer_cast(y),
                        thrust::device_pointer_cast(y),
                        gemv_functor3<T>(alpha));
    }
    else // adjoint
    {
      // temp = Sigma^{1/2} * x
      thrust::transform(Sigma_.begin(), Sigma_.end(),
                        thrust::device_pointer_cast(x),
                        temp_.begin(),
                        gemv_functor1<T>());

      // y = (beta / (alpha * Tau^{1/2})) * y
      thrust::transform(Tau_.begin(), Tau_.end(),
                        thrust::device_pointer_cast(y),
                        thrust::device_pointer_cast(y),
                        gemv_functor2<T>(alpha, beta));

      // y += A^T * temp
      linop_->EvalAdjoint(y, temp_, 1);

      // y = alpha * Tau^{1/2} * y
      thrust::transform(Tau_.begin(), Tau_.end(),
                        thrust::device_pointer_cast(y),
                        thrust::device_pointer_cast(y),
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
}

template<typename T>
void BackendADMM<T>::PerformIteration()
{
  // temp_ = x_proj_ - x_dual_
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

  // temp_ = z_proj_ - z_dual_
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

  // -----------------------------------------------------------
  // TODO: The iteration ends here. Reorder updates accordingly.
  // -----------------------------------------------------------

  // . temp1_ = alpha x_half_ + (1-alpha) x_proj_ + x_dual_
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.begin(),
          x_half_.begin(),
          x_proj_.begin(),
          x_dual_.begin())),
      
      thrust::make_zip_iterator(thrust::make_tuple(
          temp1_.end(),
          x_half_.end(),
          x_proj_.end(),
          x_dual_.end())),
      
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

  // TODO:
  // Minimize |Kx-d|^2 + |x|^2 for d = temp2_ with cgls Method
  // and store result in x_proj_
  // x_half_, x_dual_, z_proj_, z_half_, z_dual_ can be used as temp variables (for cg method)
  // add variable to warm-start cg?
  // why y0 - Ax0 as rhs in pogs?!
  
  // x_proj = Tau^{1/2} x_proj + temp1_
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
  
  // x_dual_ = temp1_ - x_proj_
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          x_dual_.begin(),
          temp1_.begin(),
          x_proj_.begin())),

      thrust::make_zip_iterator(thrust::make_tuple(
          x_dual_.end(),
          temp1_.end(),
          x_proj_.end())),

      difference_functor<T>());

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

  iteration_++;

  // TODO: move this after x^{k+1/2} and z^{k+1/2} update steps
  // compute residuals every "opts_.residual_iter" iterations and
  // adapt stepsizes for residual base adaptive schemes
  if(iteration_ == 0 || (iteration_ % opts_.residual_iter) == 0)
  {
    // TODO: compute dual residual |K^T y + w|^2 and norm |w|^2
    
    // TODO: compute primal residual |Kx - z|^2 and norm |z|^2

    // TODO: fill variables, adapt stepsizes, rescale tilde variables
  }
}

template<typename T>
void BackendADMM<T>::Release()
{
}

template<typename T>
void BackendADMM<T>::current_solution(
  vector<T>& primal, vector<T>& dual)
{
  // TODO: compute y and fill primal/dual
}

template<typename T>
void BackendADMM<T>::current_solution(
  vector<T>& primal_x,
  vector<T>& primal_z,
  vector<T>& dual_y,
  vector<T>& dual_w)
{
  // TODO: compute y,w and fill primal/dual
}

template<typename T>
size_t BackendADMM<T>::gpu_mem_amount() const
{
  size_t m = this->problem_->nrows();
  size_t n = this->problem_->ncols();

  return 4 * (n + m) * sizeof(T);
}

// Explicit template instantiation
template class BackendADMM<float>;
template class BackendADMM<double>;

} // namespace prost