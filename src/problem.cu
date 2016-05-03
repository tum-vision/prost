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
#include <random>
#include <thrust/transform_reduce.h>

#include "prost/problem.hpp"
#include "prost/linop/linearoperator.hpp"
#include "prost/linop/dual_linearoperator.hpp"
#include "prost/prox/prox.hpp"
#include "prost/prox/prox_zero.hpp"
#include "prost/prox/prox_separable_sum.hpp"
#include "prost/exception.hpp"

namespace prost {

/// \brief Used for sorting prox operators according to their starting index.
template<typename T>
struct ProxCompare {
  bool operator()(std::shared_ptr<Prox<T> > const& left, std::shared_ptr<Prox<T> > const& right) {
    if(left->index() < right->index())
      return true;

    return false;
  }
};

/// \brief Checks whether the whole domain is covered by prox operators.
template<typename T>
void CheckDomainProx(const typename Problem<T>::ProxList& proxs, size_t n, const std::string& name)
{
  size_t num_proxs = proxs.size();

  if(0 == num_proxs)
    return;

  typename Problem<T>::ProxList sorted_proxs = proxs;
  std::sort(sorted_proxs.begin(), sorted_proxs.end(), ProxCompare<T>());

  for(size_t i = 0; i < num_proxs - 1; i++) {
    
    if(sorted_proxs[i]->end() != (sorted_proxs[i + 1]->index() - 1))
    {
      stringstream ss;

      ss << name << " (CheckDomainProx): Prox operators are overlapping: [";
      ss << sorted_proxs[i]->index() << ", " << sorted_proxs[i]->end() << "] and [";
      ss << sorted_proxs[i + 1]->index() << ", " << sorted_proxs[i + 1]->end() << "]." << endl;
      throw Exception(ss.str());
    }
  }

  if(sorted_proxs[num_proxs - 1]->end() != (n - 1)) {
    stringstream ss;

    if(sorted_proxs[num_proxs - 1]->end() < (n - 1)) 
    {
      ss << name << " (CheckDomainProx): Last prox operator ends too early: [";
      ss << sorted_proxs[num_proxs - 1]->index() << ", " << sorted_proxs[num_proxs - 1]->end() << "], end = ";
      ss << n - 1 << "." << endl;
      throw Exception(ss.str());
    }
    else
    {
      ss << name << " (CheckDomainProx): Last prox operator ends after the domain: [";
      ss << sorted_proxs[num_proxs - 1]->index() << ", " << sorted_proxs[num_proxs - 1]->end() << "], end = ";
      ss << n - 1 << "." << endl;
      throw Exception(ss.str());
    }
  }
}

/// \brief Fills up domain with zero prox operators
template<typename T>
void AddZeroProx(typename Problem<T>::ProxList& proxs, size_t n, const std::string& name)
{
  size_t num_proxs = proxs.size();
  
  //std::cout << name << "," << num_proxs << std::endl;

  if(0 == num_proxs)
    return;
  
  typename Problem<T>::ProxList sorted_proxs = proxs;
  std::sort(sorted_proxs.begin(), sorted_proxs.end(), ProxCompare<T>());

/*
  for(size_t i = 0; i < num_proxs; i++) 
    std::cout << sorted_proxs[i]->index() << ", " << sorted_proxs[i]->end() << std::endl;
*/

  if(sorted_proxs[0]->index() > 0)
  {
//    std::cout << name << ": added zero prox [" << 0 << ", " << sorted_proxs[0]->index() << "]." << std::endl;
    proxs.push_back(shared_ptr<Prox<T>>(new ProxZero<T>(0, sorted_proxs[0]->index())));
  }

  for(size_t i = 0; i < num_proxs - 1; i++) 
  {
    if(sorted_proxs[i]->end() < (sorted_proxs[i + 1]->index() - 1))
    {
      size_t prox_start = sorted_proxs[i]->end() + 1;
      size_t prox_size = sorted_proxs[i + 1]->index() - sorted_proxs[i]->end() - 1;

      // fill with zero prox
//      std::cout << name << ": added zero prox [" << prox_start << ", " << prox_start + prox_size << "]." << std::endl;
      proxs.push_back(shared_ptr<Prox<T>>(new ProxZero<T>(prox_start, prox_size)));
    }
    else if(sorted_proxs[i]->end() < (sorted_proxs[i + 1]->index() - 1))
    {
      stringstream ss;

      ss << name << " (AddZeroProx): Prox operators are overlapping: [";
      ss << sorted_proxs[i]->index() << ", " << sorted_proxs[i]->end() << "] and [";
      ss << sorted_proxs[i + 1]->index() << ", " << sorted_proxs[i + 1]->end() << "]." << endl;
      throw Exception(ss.str());
    }
  }

  if(sorted_proxs[num_proxs - 1]->end() != (n - 1)) {
    stringstream ss;

    if(sorted_proxs[num_proxs - 1]->end() < (n - 1)) 
    {
      size_t prox_start = sorted_proxs[num_proxs - 1]->end() + 1;
      size_t prox_size = (n - 1) - sorted_proxs[num_proxs - 1]->end();

      // fill with zero prox
//      std::cout << name << ": added zero prox [" << prox_start << ", " << prox_start + prox_size << "]." << std::endl;
      proxs.push_back(shared_ptr<Prox<T>>(new ProxZero<T>(prox_start, prox_size)));
    }
    else
    {
      ss << name << " (AddZeroProx): Last prox operator ends after the domain: [";
      ss << sorted_proxs[num_proxs - 1]->index() << ", " << sorted_proxs[num_proxs - 1]->end() << "], end = ";
      ss << n - 1 << "." << endl;
      throw Exception(ss.str());
    }
  }
}

template<typename T>
Problem<T>::Problem() : linop_(new LinearOperator<T>()) { }

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
  {
    //prox_f_.push_back(shared_ptr<Prox<T>>(new ProxZero<T>(0, nrows_)));
    throw Exception("No proximal operator for f or fstar specified.");
  }

  if(prox_g_.empty() && prox_gstar_.empty())
  {
    //prox_g_.push_back(shared_ptr<Prox<T>>(new ProxZero<T>(0, ncols_)));

    throw Exception("No proximal operator for g or gstar specified.");
  }

  if(!prox_f_.empty() && !prox_fstar_.empty())
    throw Exception("Proximal operator for f AND fstar specified. Only set one!");

  if(!prox_g_.empty() && !prox_gstar_.empty())
    throw Exception("Proximal operator for g AND gstar specified. Only set one!");

  // Set zero prox where prox operators are not specified.
  if(!prox_f_.empty()) 
    AddZeroProx<T>(prox_f_, nrows_, "prox_f");

  if(!prox_g_.empty()) 
    AddZeroProx<T>(prox_g_, ncols_, "prox_g");

  if(!prox_fstar_.empty()) 
    AddZeroProx<T>(prox_fstar_, nrows_, "prox_fstar");

  if(!prox_gstar_.empty()) 
    AddZeroProx<T>(prox_gstar_, ncols_, "prox_gstar");

  // check if whole domain is covered by prox operators
  CheckDomainProx<T>(prox_g_, ncols_, "prox_g");
  CheckDomainProx<T>(prox_f_, nrows_, "prox_f");
  CheckDomainProx<T>(prox_gstar_, ncols_, "prox_gstar");
  CheckDomainProx<T>(prox_fstar_, nrows_, "prox_fstar");

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
    scaling_left_host_ = std::vector<T>(nrows_);
    scaling_right_host_ = std::vector<T>(ncols_);

    T value = 1;
    for(size_t row = 0; row < nrows(); row++)
    {
      T rowsum = linop_->row_sum(row, scaling_alpha_);

      if(rowsum > 0)
        value = 1. / rowsum;

      scaling_left_host_[row] = value;
    }

    for(size_t col = 0; col < ncols(); col++)
    {
      T colsum = linop_->col_sum(col, 2. - scaling_alpha_);

      if(colsum > 0)
        value = 1. / colsum;

      scaling_right_host_[col] = value;
    }
  }
  else if(scaling_type_ == Problem<T>::Scaling::kScalingIdentity)
  {
    scaling_left_host_ = std::vector<T>(nrows_, 1);
    scaling_right_host_ = std::vector<T>(ncols_, 1);
  }
  else if(scaling_type_ == Problem<T>::Scaling::kScalingCustom)
  {
    if((scaling_left_host_.size() != nrows_) || (scaling_right_host_.size() != ncols_))
      throw Exception("Preconditioners/diagonal scaling vectors do not fit the size of linear operator.");
  }

  // average preconditioners at places where prox doesn't allow diagsteps
  AveragePreconditioners(
    scaling_right_host_,
    prox_g_.empty() ? prox_gstar_ : prox_g_);

  AveragePreconditioners(
    scaling_left_host_,
    prox_f_.empty() ? prox_fstar_ : prox_f_);

  // copy to gpu
  scaling_left_.resize(nrows());
  scaling_right_.resize(ncols());

  thrust::copy(
    scaling_left_host_.begin(), 
    scaling_left_host_.end(), 
    scaling_left_.begin());

  thrust::copy(
    scaling_right_host_.begin(), 
    scaling_right_host_.end(), 
    scaling_right_.begin());

  dual_linop_ = shared_ptr<LinearOperator<T>>(new DualLinearOperator<T>(linop_));
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

template<typename T>
void Problem<T>::SetScalingCustom(
  const std::vector<T>& left, 
  const std::vector<T>& right)
{
  scaling_type_ = Problem<T>::Scaling::kScalingCustom;

  scaling_left_host_ = std::vector<T>(left.size());
  scaling_right_host_ = std::vector<T>(right.size());

  std::transform(
    left.begin(), 
    left.end(),
    scaling_left_host_.begin(),
    [](T x) { return x * x; } );

  std::transform(
    right.begin(), 
    right.end(),
    scaling_right_host_.begin(),
    [](T x) { return x * x; } );
}

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
struct normest_multiplies_sqrt : public thrust::binary_function<T, T, T>
{
  __host__ __device__
  T operator()(const T& a, const T& b) const
  {
    return sqrt(a)*b;
  }
};

template<typename T>
T Problem<T>::normest(T tol, int max_iters)
{
  thrust::device_vector<T> x(ncols()), Ax(nrows());
  thrust::device_vector<T> x_temp(ncols()), Ax_temp(nrows());

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
      x_temp.begin(), 
      normest_multiplies_sqrt<T>());

    linop_->Eval(Ax_temp, x_temp);

    thrust::transform(
      scaling_left_.begin(), 
      scaling_left_.end(),
      Ax_temp.begin(), 
      Ax.begin(), 
      normest_multiplies_sqrt<T>());

    T norm_Ax = std::sqrt( thrust::transform_reduce(
        Ax.begin(), 
        Ax.end(), 
        normest_square<T>(), 
        static_cast<T>(0), 
        thrust::plus<T>()) ); 

    thrust::transform(
      scaling_left_.begin(), 
      scaling_left_.end(),
      Ax.begin(), 
      Ax_temp.begin(), 
      normest_multiplies_sqrt<T>());

    linop_->EvalAdjoint(x_temp, Ax_temp);

    thrust::transform(
      scaling_right_.begin(), 
      scaling_right_.end(),
      x_temp.begin(), 
      x.begin(), 
      normest_multiplies_sqrt<T>());

    T norm_x = std::sqrt( thrust::transform_reduce(
        x.begin(), 
        x.end(), 
        normest_square<T>(), 
        static_cast<T>(0), 
        thrust::plus<T>()) ); 

    norm = norm_x / norm_Ax;

    if(std::abs(norm_prev - norm) < tol * norm)
    {
      break;
    }

    thrust::transform(x.begin(), x.end(), x.begin(), normest_divide<T>(norm_x));
  }

  return norm;
}

template<typename T>
void Problem<T>::AveragePreconditioners(
    std::vector<T>& precond,
    const ProxList& prox)
{
  std::vector<std::tuple<size_t, size_t, size_t> > idx_cnt_std;
  idx_cnt_std.reserve(precond.size());

  // compute places where to average
  for(auto& p : prox)
  {
    if(!p->diagsteps())
    {
      p->get_separable_structure(idx_cnt_std);
    }
  }

  // perform averaging
  for(auto& ics : idx_cnt_std)
  {
    size_t idx = std::get<0>(ics);
    size_t cnt = std::get<1>(ics);
    size_t std = std::get<2>(ics);

    // compute average
    T avg = 0;
    for(size_t c = 0; c < cnt; c++)
      avg += precond[idx + c * std];
    avg /= static_cast<T>(cnt);

    // fill values
    for(size_t c = 0; c < cnt; c++)
      precond[idx + c * std] = avg;
  }
}

template<typename T>
void Problem<T>::Dualize()
{
  prox_g_.swap(prox_fstar_);
  prox_gstar_.swap(prox_f_);
  std::swap(nrows_, ncols_);
  std::swap(linop_, dual_linop_);
  scaling_left_.swap(scaling_right_); // TODO: does this work?
  std::swap(scaling_left_host_, scaling_right_host_);
}

// Explicit template instantiation
template class Problem<float>;
template class Problem<double>;

} // namespace prost
