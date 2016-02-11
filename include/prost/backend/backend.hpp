#ifndef PROST_BACKEND_HPP_
#define PROST_BACKEND_HPP_

#include "prost/common.hpp"
#include "prost/solver.hpp"

namespace prost {

template<typename T> class Problem;

///
/// \brief Abstract base class for primal-dual algorithms solving graph form 
///        problems.
/// 
template<typename T>
class Backend {
public:
  Backend() {}
  virtual ~Backend() {}

  void SetProblem(shared_ptr<Problem<T> > problem) { problem_ = problem; }
  void SetOptions(const typename Solver<T>::Options& opts) { solver_opts_ = opts; }

  virtual void Initialize() = 0;
  virtual void PerformIteration() = 0;
  virtual void Release() = 0;

  /// \brief Copies current primal dual solution pair (x,y) to the host.
  virtual void current_solution(vector<T>& primal_sol, vector<T>& dual_sol) const = 0;

  /// \brief Returns norm of the primal residual |Ax - z|.
  virtual T primal_residual() const { return primal_residual_; }

  /// \brief Returns norm of the dual residual |A^T y + w|.
  virtual T dual_residual() const { return dual_residual_; }

  /// \brief Returns norm of the primal variable "z", used for stopping criterion.
  virtual T primal_var_norm() const  { return primal_var_norm_; }

  /// \brief Returns norm of the dual variable "w", used for stopping criterion.
  virtual T dual_var_norm() const { return dual_var_norm_; }

  // returns amount of gpu memory required in bytes
  virtual size_t gpu_mem_amount() const = 0;

protected:
  shared_ptr<Problem<T> > problem_;

  typename Solver<T>::Options solver_opts_;

  /// \brief Norm of the primal variable z = Kx.
  T primal_var_norm_;

  /// \brief Norm of the dual variable w = -K^T y.
  T dual_var_norm_;

  /// \brief Size of primal residual |Kx - z|.
  T primal_residual_;

  /// \brief Size of dual residual |K^T y + w|
  T dual_residual_;
};

} // namespace prost

#endif // PROST_BACKEND_HPP_
