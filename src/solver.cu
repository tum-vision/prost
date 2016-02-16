#include "prost/solver.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <locale>
#include <list>
#include <sstream>

#include "prost/backend/backend.hpp"
#include "prost/common.hpp"
#include "prost/problem.hpp"
#include "prost/exception.hpp"

namespace prost {

using std::cout;
using std::endl;

template<typename T>
Solver<T>::Solver(std::shared_ptr<Problem<T> > problem, std::shared_ptr<Backend<T> > backend) 
    : problem_(problem), backend_(backend)
{
}

template<typename T> 
void Solver<T>::SetOptions(const typename Solver<T>::Options& opts) {
  opts_ = opts;
}

template<typename T>
void Solver<T>::SetStoppingCallback(const typename Solver<T>::StoppingCallback& cb) {
  stopping_cb_ = cb;
}

template<typename T>
void Solver<T>::SetIntermCallback(const typename Solver<T>::IntermCallback& cb) {
  interm_cb_ = cb;
}

template<typename T>
void Solver<T>::Initialize() {
  try
  {
    problem_->Initialize();
  }
  catch(Exception& e)
  {
    stringstream ss;
    ss << "Failed to initialize the problem. Reason: " << e.what();
    throw Exception(ss.str());
  }
  
  if(opts_.solve_dual_problem)
  {
    problem_->Dualize();
    opts_.x0.swap(opts_.y0);
  }
  
  try
  {
    backend_->SetProblem(problem_);
    backend_->SetOptions(opts_);
    backend_->Initialize();
  }
  catch(Exception& e)
  {
    stringstream ss;
    ss << "Failed to initialize the backend. Reason: " << e.what();
    throw Exception(ss.str());
  }

  if (opts_.verbose)
  {
    size_t mem = problem_->gpu_mem_amount() + backend_->gpu_mem_amount();
    size_t mem_avail, mem_total;
    cudaMemGetInfo(&mem_avail, &mem_total);
    mem_avail /= 1024 * 1024;
    mem_total /= 1024 * 1024;

    std::cout << "# primal variables: " << problem_->ncols() << std::endl;
    std::cout << "# dual variables: " << problem_->nrows() << std::endl;
    std::cout << "Memory requirements: " << mem / (1024 * 1024) << "MB (" << mem_avail << "/" << mem_total << "MB available)." << std::endl;
  }

  cur_primal_sol_.resize( problem_->ncols() );
  cur_dual_sol_.resize( problem_->nrows() );
}

template<typename T>
typename Solver<T>::ConvergenceResult Solver<T>::Solve() {
  typename Solver<T>::ConvergenceResult result =
      Solver<T>::ConvergenceResult::kStoppedMaxIters;
  
  // iterations to display
  std::list<double> cb_iters =
      linspace(0, opts_.max_iters - 1, opts_.num_cback_calls);
  
  for(int i = 0; i < opts_.max_iters; i++) {    
    backend_->PerformIteration();

    // check if solver has converged
    T primal_res = backend_->primal_residual();
    T dual_res = backend_->dual_residual();
    T eps_pri = backend_->eps_primal();
    T eps_dua = backend_->eps_dual();

    bool is_converged = false;
    bool is_stopped = stopping_cb_();

    if((primal_res < eps_pri) && (dual_res < eps_dua))
      is_converged = true;

    // check if we should run the intermediate solution callback this iteration
    if(i >= cb_iters.front() || is_converged || is_stopped) {
      backend_->current_solution(cur_primal_sol_, cur_dual_sol_);
 
      if(opts_.verbose) {
        int digits = std::floor(std::log10( (double) opts_.max_iters )) + 1;
        cout << "It " << std::setw(digits) << (i + 1) << ": " << std::scientific;
        cout << "Feas_p=" << std::setprecision(2) << primal_res;
        cout << ", Eps_p=" << std::setprecision(2) << eps_pri;
        cout << ", Feas_d=" << std::setprecision(2) << dual_res;
        cout << ", Eps_d=" << std::setprecision(2) << eps_dua << "; ";
      }

      // MATLAB callback
      if(opts_.solve_dual_problem)
        interm_cb_(i + 1, cur_dual_sol_, cur_primal_sol_);
      else
        interm_cb_(i + 1, cur_primal_sol_, cur_dual_sol_);
      
      cb_iters.pop_front();
    }

    if(is_stopped) {
      if(opts_.verbose)
        std::cout << "Stopped by user." << std::endl;

      result = Solver<T>::ConvergenceResult::kStoppedUser;
      break;
    }

    if(is_converged) {
      if(opts_.verbose)
        std::cout << "Reached convergence tolerance." << std::endl;

      result = Solver<T>::ConvergenceResult::kConverged;
      break;
    }
  }

  // restore original problem
  if(opts_.solve_dual_problem)
  {
    problem_->Dualize();
    opts_.x0.swap(opts_.y0);
  }
  
  if(opts_.verbose && (result == Solver<T>::ConvergenceResult::kStoppedMaxIters))
    std::cout << "Reached maximum iterations." << std::endl;

  return result;
}

template<typename T>
void Solver<T>::Release() {
  problem_->Release();
  backend_->Release();
}

template<typename T>
const vector<T>& Solver<T>::cur_primal_sol() const
{
  if(opts_.solve_dual_problem)
    return cur_dual_sol_;

  return cur_primal_sol_;
}

template<typename T>
const vector<T>& Solver<T>::cur_dual_sol() const
{
  if(opts_.solve_dual_problem)
    return cur_primal_sol_;

  return cur_dual_sol_;
}

// Explicit template instantiation
template class Solver<float>;
template class Solver<double>;

} // namespace prost
