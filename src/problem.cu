#include "problem.hpp"

#include <algorithm>

#include "prox/prox.hpp"
#include "linop/linearoperator.hpp"
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
void Problem<T>::AddBlock(Block<T> *block)
{
  linop_->AddBlock(std::shared_ptr<Block<T> >(block));
}

template<typename T>
void Problem<T>::AddProx_g(Prox<T> *prox)
{
  prox_g_.push_back(std::shared_ptr<Prox<T> >(prox));
}

template<typename T>
void Problem<T>::AddProx_f(Prox<T> *prox)
{
  prox_f_.push_back(std::shared_ptr<Prox<T> >(prox));
}

template<typename T>
void Problem<T>::AddProx_gstar(Prox<T> *prox)
{
  prox_gstar_.push_back(std::shared_ptr<Prox<T> >(prox));
}

template<typename T>
void Problem<T>::AddProx_fstar(Prox<T> *prox)
{
  prox_fstar_.push_back(std::shared_ptr<Prox<T> >(prox));
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
    throw new Exception("No proximal operator for f or fstar specified.");

  if(prox_g_.empty() && prox_gstar_.empty())
    throw new Exception("No proximal operator for g or gstar specified.");

  // check if whole domain is covered by prox operators
  if(!CheckDomainProx<T>(prox_g_, ncols_)) 
    throw new Exception("prox_g does not cover the whole domain!");

  if(!CheckDomainProx<T>(prox_f_, nrows_)) 
    throw new Exception("prox_f does not cover the whole domain!");
   
  if(!CheckDomainProx<T>(prox_gstar_, ncols_)) 
    throw new Exception("prox_gstar does not cover the whole domain!");

  if(!CheckDomainProx<T>(prox_fstar_, nrows_)) 
    throw new Exception("prox_fstar does not cover the whole domain!");

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
    throw new Exception("Alpha scaling is not implemented yet.");
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
      throw new Exception("Preconditioners/diagonal scaling vectors do not fit the size of linear operator.");
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

// Explicit template instantiation
template class Problem<float>;
template class Problem<double>;
