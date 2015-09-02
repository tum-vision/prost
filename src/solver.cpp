#include "solver.hpp"

#include <iostream>
#include "solver_backend.hpp"
#include "solver_backend_pdhg.hpp"
#include "util/util.hpp"

Solver::Solver()
    : backend_(NULL), h_primal_(NULL), h_dual_(NULL) { 
}

Solver::~Solver() {
}

void Solver::SetMatrix(SparseMatrix<real>* mat) {
  problem_.mat = mat;
  problem_.nrows = mat->nrows();
  problem_.ncols = mat->ncols();
}

void Solver::AddProx_g(Prox* prox) {
  problem_.prox_g.push_back(prox);
}

void Solver::AddProx_hc(Prox* prox) {
  problem_.prox_hc.push_back(prox);
}

void Solver::SetOptions(const SolverOptions& opts) {
  opts_ = opts;
}

void Solver::SetCallback(SolverCallbackPtr cb) {
  callback_ = cb;
}

bool Solver::Initialize() {
  h_primal_ = new real[problem_.ncols];
  h_dual_ = new real[problem_.nrows];

  // create preconditioners
  problem_.precond = new Preconditioner(problem_.mat);
  switch(opts_.precond)
  {
    case kPrecondScalar:
      problem_.precond->ComputeScalar();
      break;

    case kPrecondAlpha:
      return false; // not implemented yet

    case kPrecondEquil:
      return false; // not implemented yet
  }

  // create backend
  switch(opts_.backend)
  {
    case kBackendPDHG:
      backend_ = new SolverBackendPDHG(problem_, opts_);
      break;

    default:
      return false;
  }

  // calculate memory requirements
  size_t gpu_mem = 0;
  for(int i = 0; i < problem_.prox_g.size(); ++i)
    gpu_mem += problem_.prox_g[i]->gpu_mem_amount();

  for(int i = 0; i < problem_.prox_hc.size(); ++i)
    gpu_mem += problem_.prox_hc[i]->gpu_mem_amount();

  gpu_mem += backend_->gpu_mem_amount();
  gpu_mem += problem_.mat->gpu_mem_amount();
  gpu_mem += problem_.precond->gpu_mem_amount();

  std::cout << "Total GPU memory required: " << gpu_mem / (1024 * 1024) << "MB." << std::endl;

  size_t gpu_mem_avail, gpu_mem_free;
  cudaMemGetInfo(&gpu_mem_free, &gpu_mem_avail);
  
  if(gpu_mem > gpu_mem_avail) {
    std::cout << "Out of memory! Only " << gpu_mem_avail / (1024 * 1024) << "MB available." << std::endl;
    return false;
  }

  if(!backend_->Initialize())
    return false;

  return true;
}

void Solver::Solve() {
  // iterations to display
  std::list<double> cb_iters =
      linspace(0, opts_.max_iters - 1, opts_.cb_iters);
  
  for(int i = 0; i < opts_.max_iters; i++) {    
    backend_->PerformIteration();
    bool is_converged = backend_->converged();

    // check if we should run the callback this iteration
    if(i >= cb_iters.front()) {
      backend_->iterates(h_primal_, h_dual_);
      callback_(i + 1, h_primal_, h_dual_, false);
      cb_iters.pop_front();
    }

    if(is_converged)
      break;
  }
}

void Solver::Release() {
  backend_->Release();
  
  delete [] h_primal_;
  delete [] h_dual_;
  delete problem_.precond;
}
