#include "problem.hpp"

#include <algorithm>
#include <random>
#include <thrust/transform_reduce.h>

#include "linop/linearoperator.hpp"
#include "prox/prox.hpp"
#include "exception.hpp"

// used for sorting prox operators according to their starting index
template<typename T>
struct ProxCompare {
  bool operator()(std::shared_ptr<Prox<T> > const& left, std::shared_ptr<Prox<T> > const& right) {
    if(left->index() < right->index())
      return true;

    return false;
  }
};

/**
 * @brief Checks whether the whole domain is covered by prox operators.
 */
template<typename T>
bool CheckDomainProx(const typename Problem<T>::ProxList& proxs, size_t n) 
{
  size_t num_proxs = proxs.size();

  if(0 == num_proxs)
    return true;

  typename Problem<T>::ProxList sorted_proxs = proxs;
  std::sort(sorted_proxs.begin(), sorted_proxs.end(), ProxCompare<T>());

  for(size_t i = 0; i < num_proxs - 1; i++) {
    
    if(sorted_proxs[i]->end() != (sorted_proxs[i + 1]->index() - 1))
    {
      return false;
    }
  }

  if(sorted_proxs[num_proxs - 1]->end() != (n - 1)) {
    return false;
  }

  return true;
}

template<typename T>
Problem<T>::Problem()
  : linop_(new LinearOperator<T>())
{
}

template<typename T>
Problem<T>::~Problem()
{
}

template<typename T>
void Problem<T>::AddBlock(std::shared_ptr<Block<T> > block)
{
  linop_->AddBlock(block);
}

template<typename T>
void Problem<T>::AddProx_g(std::shared_ptr<Prox<T> > prox)
{
  prox_g_.push_back(prox);
}

template<typename T>
void Problem<T>::AddProx_f(std::shared_ptr<Prox<T> > prox)
{
  prox_f_.push_back(prox);
}

template<typename T>
void Problem<T>::AddProx_gstar(std::shared_ptr<Prox<T> > prox)
{
  prox_gstar_.push_back(prox);
}

template<typename T>
void Problem<T>::AddProx_fstar(std::shared_ptr<Prox<T> > prox)
{
  prox_fstar_.push_back(prox);
}

// builds the linear operator and checks if prox cover the 
// whole domain
template<typename T>
void Problem<T>::Initialize()
{
  linop_->Initialize();
  nrows_ = linop_->nrows();
  ncols_ = linop_->ncols();

  if(prox_f_.empty() && prox_fstar_.empty())
    throw Exception("No proximal operator for f or fstar specified.");

  if(prox_g_.empty() && prox_gstar_.empty())
    throw Exception("No proximal operator for g or gstar specified.");

  // check if whole domain is covered by prox operators
  if(!CheckDomainProx<T>(prox_g_, ncols_)) 
    throw Exception("prox_g does not cover the whole domain!");

  if(!CheckDomainProx<T>(prox_f_, nrows_)) 
    throw Exception("prox_f does not cover the whole domain!");
   
  if(!CheckDomainProx<T>(prox_gstar_, ncols_)) 
    throw Exception("prox_gstar does not cover the whole domain!");

  if(!CheckDomainProx<T>(prox_fstar_, nrows_)) 
    throw Exception("prox_fstar does not cover the whole domain!");

  // Init Proxs
  for(auto& prox : prox_f_) 
    prox->Initialize();

  for(auto& prox : prox_fstar_) 
    prox->Initialize();

  for(auto& prox : prox_g_) 
    prox->Initialize();

  for(auto& prox : prox_gstar_) 
    prox->Initialize(); 

  // Init Scaling
  if(scaling_type_ == Problem<T>::Scaling::kScalingAlpha)
  {
    scaling_left_.resize(nrows());
    scaling_right_.resize(ncols());

    std::vector<T> left, right;
    left.resize(nrows());
    right.resize(ncols());

    // TODO: average step sizes for points where prox doesn't allow diagsteps
    for(size_t row = 0; row < nrows(); row++)
    {
      T rowsum = linop_->row_sum(row, scaling_alpha_);
      left[row] = 1. / ((rowsum > 0) ? rowsum : 1.);
    }

    for(size_t col = 0; col < ncols(); col++)
    {
      T colsum = linop_->col_sum(col, scaling_alpha_);
      right[col] = 1. / ((colsum > 0) ? colsum : 1.);
    }

    thrust::copy(left.begin(), left.end(), scaling_left_.begin());
    thrust::copy(right.begin(), right.end(), scaling_right_.begin());
  }
  else if(scaling_type_ == Problem<T>::Scaling::kScalingIdentity)
  {
    scaling_left_ = thrust::device_vector<T>(nrows_, 1);
    scaling_right_ = thrust::device_vector<T>(ncols_, 1);
  }
  else if(scaling_type_ == Problem<T>::Scaling::kScalingCustom)
  {
    scaling_left_ = thrust::device_vector<T>(scaling_left_custom_.begin(), scaling_left_custom_.end());
    scaling_right_ = thrust::device_vector<T>(scaling_right_custom_.begin(), scaling_right_custom_.end());

    if((scaling_left_custom_.size() != nrows_) || (scaling_right_custom_.size() != ncols_))
      throw Exception("Preconditioners/diagonal scaling vectors do not fit the size of linear operator.");
  }
}

template<typename T>
void Problem<T>::Release()
{
  linop_->Release();

  for(auto& prox : prox_f_) 
    prox->Release();

  for(auto& prox : prox_fstar_) 
    prox->Release();

  for(auto& prox : prox_g_) 
    prox->Release();

  for(auto& prox : prox_gstar_) 
    prox->Release();
}

// sets a predefined problem scaling
template<typename T>
void Problem<T>::SetScalingCustom(
  const std::vector<T>& left, 
  const std::vector<T>& right)
{
  scaling_type_ = Problem<T>::Scaling::kScalingCustom;
  scaling_left_custom_ = left;
  scaling_right_custom_ = right;
}

// computes a scaling using the Diagonal Preconditioners
// proposed in Pock, Chambolle ICCV '11.
template<typename T>
void Problem<T>::SetScalingAlpha(T alpha)
{
  scaling_type_ = Problem<T>::Scaling::kScalingAlpha;
  scaling_alpha_ = alpha;
}

template<typename T>
void Problem<T>::SetScalingIdentity()
{
  scaling_type_ = Problem<T>::Scaling::kScalingIdentity;
}

template<typename T>
size_t Problem<T>::gpu_mem_amount() const
{
  size_t mem = 0;

  for(auto& p : prox_f_) mem += p->gpu_mem_amount();
  for(auto& p : prox_g_) mem += p->gpu_mem_amount();
  for(auto& p : prox_fstar_) mem += p->gpu_mem_amount();
  for(auto& p : prox_gstar_) mem += p->gpu_mem_amount();
  mem += linop_->gpu_mem_amount();
  mem += sizeof(T) * (nrows() + ncols());

  return mem;
}

template<typename T>
struct normest_square
{
  __host__ __device__
  T operator()(const T& x) const 
  { 
    return x * x;
  }
};

template<typename T>
struct normest_divide
{
  normest_divide(T fac) : fac_(fac) { }

  __host__ __device__
  T operator()(const T& x) const 
  { 
    return x / fac_;
  }

  T fac_;
};

template<typename T>
struct normest_multiply_square : public thrust::binary_function<T, T, T>
{
  __host__ __device__
  T operator()(const T& a, const T& b) const
  {
    return a*a*b;
  }
};

template<typename T>
T Problem<T>::normest(T tol, int max_iters)
{
  thrust::device_vector<T> x(ncols()), Ax(nrows());

  std::vector<T> x_host(ncols());
  std::generate(x_host.begin(), x_host.end(), []{ return (T)std::rand() / (T)RAND_MAX; } );
  thrust::copy(x_host.begin(), x_host.end(), x.begin());

  T norm = 0, norm_prev;
  for(int i = 0; i < max_iters; i++)
  {
    norm_prev = norm;

    thrust::transform(
      scaling_right_.begin(), 
      scaling_right_.end(),
      x.begin(), 
      x.begin(), 
      thrust::multiplies<T>());

    linop_->Eval(Ax, x);

    thrust::transform(
      scaling_left_.begin(), 
      scaling_left_.end(),
      Ax.begin(), 
      Ax.begin(), 
      normest_multiply_square<T>());

    linop_->EvalAdjoint(x, Ax);

    thrust::transform(
      scaling_right_.begin(), 
      scaling_right_.end(),
      x.begin(), 
      x.begin(), 
      thrust::multiplies<T>());

    T norm_x = std::sqrt( thrust::transform_reduce(
        x.begin(), 
        x.end(), 
        normest_square<T>(), 
        static_cast<T>(0), 
        thrust::plus<T>()) ); 

    T norm_Ax = std::sqrt( thrust::transform_reduce(
        Ax.begin(), 
        Ax.end(), 
        normest_square<T>(), 
        static_cast<T>(0), 
        thrust::plus<T>()) ); 

    thrust::transform(x.begin(), x.end(), x.begin(), normest_divide<T>(norm_x));
    norm = norm_x / norm_Ax;

    if(std::abs(norm_prev - norm) < tol * norm)
      break;
  }

  return norm;
}


// Explicit template instantiation
template class Problem<float>;
template class Problem<double>;
