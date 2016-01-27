#include "backend/backend_pdhg.hpp"

#include "problem.hpp"

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
BackendPDHG<T>::SetStepsizeCallback(const typename BackendPDHG<T>::StepsizeCallback& cb)
{
  stepsize_cb_ = cb; 
}

template<typename T>
void 
BackendPDHG<T>::Initialize()
{
}

template<typename T>
void 
BackendPDHG<T>::PerformIteration()
{
}

template<typename T>
void 
BackendPDHG<T>::Release()
{
}

template<typename T>
void 
BackendPDHG<T>::current_solution(std::vector<T>& primal, std::vector<T>& dual) const
{
}

template<typename T>
T 
BackendPDHG<T>::primal_var_norm() const
{
  return 0;
}

template<typename T>
T 
BackendPDHG<T>::dual_var_norm() const
{
  return 0;
}

template<typename T>
T 
BackendPDHG<T>::primal_residual() const
{
  return 0;
}

template<typename T>
T 
BackendPDHG<T>::dual_residual() const
{
  return 0;
}

template<typename T>
size_t 
BackendPDHG<T>::gpu_mem_amount() const
{
  return 0;
}

// Explicit template instantiation
template class BackendPDHG<float>;
template class BackendPDHG<double>;
