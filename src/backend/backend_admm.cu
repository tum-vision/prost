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

  // allocate variables
  try
  {
    xz_half_.resize(n + m, 0);
    xz_proj_.resize(n + m, 0);
    xz_dual_.resize(n + m, 0);
  }
  catch(std::bad_alloc& e)
  {
    std::stringstream ss;
    ss << "Out of memory: " << e.what();
    throw Exception(ss.str());
  }

  x_half_ = xz_half_.data(); z_half_ = xz_half_.data() + n;
  x_proj_ = xz_proj_.data(); z_proj_ = xz_proj_.data() + n;
  x_dual_ = xz_dual_.data(); z_dual_ = xz_dual_.data() + n;
}

template<typename T>
void BackendADMM<T>::PerformIteration()
{
  // xz_half_ = xz_proj_ - xz_dual_

  // apply prox to x_half_ and z_half_

  // project xz_half_ + xz_dual_ onto graph, store result in xz_proj_

  // xz_dual_ = xz_dual_ + xz_half_ - xz_proj_

  // update residuals and step-size, check stopping criteria, etc.
}

template<typename T>
void BackendADMM<T>::Release()
{
}

template<typename T>
void BackendADMM<T>::current_solution(
  vector<T>& primal, vector<T>& dual)
{
  // what are x,y?
}

template<typename T>
void BackendADMM<T>::current_solution(
  vector<T>& primal_x,
  vector<T>& primal_z,
  vector<T>& dual_y,
  vector<T>& dual_w)
{
  // what are x,z,y,w?
}

template<typename T>
size_t BackendADMM<T>::gpu_mem_amount() const
{
  size_t m = this->problem_->nrows();
  size_t n = this->problem_->ncols();

  return 3 * (n + m) * sizeof(T);
}

// Explicit template instantiation
template class BackendADMM<float>;
template class BackendADMM<double>;

} // namespace prost