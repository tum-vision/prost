#include "solver/solver.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include "solver/solver_backend.hpp"
#include "solver/solver_backend_pdhg.hpp"
#include "util/util.hpp"

// used for sorting prox operators according to their starting index
struct ProxCompare {
  bool operator()(Prox<real>* const& left, Prox<real>* const& right) {
    if(left->index() < right->index())
      return true;

    return false;
  }
};

/**
 * @brief Checks whether the whole domain is covered by prox operators.
 */
bool CheckDomainProx(const std::vector<Prox<real> *>& proxs, int n) {
  int num_proxs = proxs.size();

  std::vector<Prox<real> *> sorted_proxs = proxs;
  std::sort(sorted_proxs.begin(), sorted_proxs.end(), ProxCompare());

  //std::cout << "num_proxs=" << num_proxs << ":" << std::endl;
  for(int i = 0; i < num_proxs - 1; i++) {
    //std::cout << sorted_proxs[i]->index() << ", " << sorted_proxs[i]->end() << std::endl;
    
    if(sorted_proxs[i]->end() != (sorted_proxs[i + 1]->index() - 1))
    {
      std::cout << "prox_index=" << i << ", (" << sorted_proxs[i]->end() << " != " << sorted_proxs[i+1]->index() - 1 << ")" << std::endl;
      return false;
    }
  }

  //std::cout << sorted_proxs[num_proxs - 1]->index() << ", " << sorted_proxs[num_proxs - 1]->end() << ", end=" << n-1 << std::endl;

  if(sorted_proxs[num_proxs - 1]->end() != (n - 1)) {
    std::cout << "prox_index=" << num_proxs - 1 << ", (" << sorted_proxs[num_proxs - 1]->end() << " != " << (n-1) << ")" << std::endl;
    
    return false;
  }

  return true;
}


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

void Solver::SetProx_g(const std::vector<Prox<real> *>& prox) {
  problem_.prox_g = prox;
}

void Solver::SetProx_hc(const std::vector<Prox<real> *>& prox) {
  problem_.prox_hc = prox;
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

  std::cout << "Creating Preconditioners...";
  std::cout.flush();
  problem_.precond = new Preconditioner(problem_.mat);
  switch(opts_.precond)
  {
    case kPrecondScalar:
      problem_.precond->ComputeScalar();
      break;

    case kPrecondAlpha:
      problem_.precond->ComputeAlpha(opts_.precond_alpha,
                                     problem_.prox_g,
                                     problem_.prox_hc);
      break;

    case kPrecondEquil:
      return false; // not implemented yet
  }
  std::cout << " done!" << std::endl;

  // create backend
  switch(opts_.backend)
  {
    case kBackendPDHG:
      backend_ = new SolverBackendPDHG(problem_, opts_);
      break;

    default:
      return false;
  }

  // check if whole domain is covered by prox operators
  if(!CheckDomainProx(problem_.prox_g, problem_.ncols)) {
    std::cout << "ERROR: prox_g does not cover the whole domain!" << std::endl;
    return false;
  }

  if(!CheckDomainProx(problem_.prox_hc, problem_.nrows)) {
    std::cout << "ERROR: prox_hc does not cover the whole domain!" << std::endl;

    return false;
  }

  for(size_t i = 0; i < problem_.prox_g.size(); i++) {
    if(!problem_.prox_g[i]->Init()) {
      std::cout << "Failed to initialized prox_g!" << std::endl;

      return false;      
    }
  }

  for(size_t i = 0; i < problem_.prox_hc.size(); i++) {
    if(!problem_.prox_hc[i]->Init()) {
      std::cout << "Failed to initialized prox_hc!" << std::endl;

      return false;      
    }
  }
  
  if(!backend_->Initialize()) {
    std::cout << "Failed to initialize the backend!" << std::endl;

    return false;
  }

  return true;
}

void Solver::gpu_mem_amount(size_t& gpu_mem_required, size_t& gpu_mem_avail, size_t& gpu_mem_free) {
  // calculate memory requirements
  gpu_mem_required = 0;
  gpu_mem_avail = 0;

  size_t gpu_mem_prox_g = 0,
      gpu_mem_prox_hc = 0,
      gpu_mem_solver = 0,
      gpu_mem_mat = 0,
      gpu_mem_precond= 0;

  for(int i = 0; i < problem_.prox_g.size(); ++i)
    gpu_mem_prox_g += problem_.prox_g[i]->gpu_mem_amount();

  for(int i = 0; i < problem_.prox_hc.size(); ++i)
    gpu_mem_prox_hc += problem_.prox_hc[i]->gpu_mem_amount();

  gpu_mem_solver += backend_->gpu_mem_amount();
  gpu_mem_mat += problem_.mat->gpu_mem_amount();
  gpu_mem_precond += problem_.precond->gpu_mem_amount();

  cudaMemGetInfo(&gpu_mem_free, &gpu_mem_avail);

  std::cout << "Memory requirements:" << std::endl;
  std::cout << "  - prox_g : " << gpu_mem_prox_g / (1024 * 1024) << "MBytes." << std::endl;
  std::cout << "  - prox_hc : " << gpu_mem_prox_hc / (1024 * 1024) << "MBytes." << std::endl;
  std::cout << "  - solver : " << gpu_mem_solver / (1024 * 1024) << "MBytes." << std::endl;
  std::cout << "  - matrix : " << gpu_mem_mat / (1024 * 1024) << "MBytes." << std::endl;
  std::cout << "  - preconds : " << gpu_mem_precond / (1024 * 1024) << "MBytes." << std::endl;

  gpu_mem_required = gpu_mem_prox_g +
                     gpu_mem_prox_hc +
                     gpu_mem_solver +
                     gpu_mem_mat +
                     gpu_mem_precond;
}

void Solver::Solve() {
  // iterations to display
  std::list<double> cb_iters =
      linspace(0, opts_.max_iters - 1, opts_.cb_iters);
  
  for(int i = 0; i < opts_.max_iters; i++) {    
    backend_->PerformIteration();
    bool is_converged = backend_->converged() && (i > 25);

    // check if we should run the callback this iteration
    if(i >= cb_iters.front() || is_converged) {
      backend_->iterates(h_primal_, h_dual_);
      callback_(i + 1, h_primal_, h_dual_, false);

      if(opts_.verbose) {
        std::cout << backend_->status();
      }
      
      cb_iters.pop_front();
    }

    if(is_converged)
      break;
  }
}

void Solver::Release() {
  backend_->Release();

  for(int i = 0; i < problem_.prox_g.size(); i++)
    delete problem_.prox_g[i];

  for(int i = 0; i < problem_.prox_hc.size(); i++)
    delete problem_.prox_hc[i];
  
  delete [] h_primal_;
  delete [] h_dual_;
  delete problem_.precond;
  delete problem_.mat;
}

std::string SolverOptions::get_string() const {
  std::stringstream ss;

  ss << "Specified solver options:" << std::endl;
  ss << " - backend:";
  if(backend == kBackendPDHG) {
    ss << " PDHG,";

    switch(adapt) {
      case kAdaptNone:
        ss << " with constant steps." << std::endl;
        break;

      case kAdaptStrong:
        ss << " accelerated version for strongly convex problems (Alg. 2). gamma = " << ad_strong.gamma << std::endl;
        break;

      case kAdaptBalance:
        ss << " residual balancing. (alpha0 = " << ad_balance.alpha0;
        ss << ", nu = " << ad_balance.nu;
        ss << ", delta = " << ad_balance.delta;
        ss << ", s = " << ad_balance.s << ")." << std::endl;
        break;

      case kAdaptConverge:
        ss << " residual converging. (delta = " << ad_converge.delta;
        ss << ", tau = " << ad_converge.tau;
        ss << std::endl;
        break;
    }
  }
  ss << " - max_iters: " << max_iters << std::endl;
  ss << " - cb_iters: " << cb_iters << std::endl;
  ss << " - tol_primal: " << tol_primal << std::endl;
  ss << " - tol_dual: " << tol_dual << std::endl;
  ss << " - verbose: " << verbose << std::endl;
  ss << " - preconditioning: ";

  switch(precond) {
    case kPrecondScalar:
      ss << "scalar." << std::endl;
      break;
    case kPrecondAlpha:
      ss << "diagonal (alpha = " << precond_alpha << ")." << std::endl;
      break;
    case kPrecondEquil:
      ss << "matrix equilibration." << std::endl;
      break;
  }
  
  return ss.str();
}
