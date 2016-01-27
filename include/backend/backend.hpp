#ifndef BACKEND_HPP_
#define BACKEND_HPP_

#include <memory>

template<typename T> class Problem;

///
/// \brief Abstract base class for primal-dual algorithms solving graph form 
///        problems.
/// 
template<typename T>
class Backend {
public:
  Backend() {}
  virtual ~Backend() { }

  void SetProblem(std::shared_ptr<Problem<T> > problem) { problem_ = problem; }

  virtual void Initialize() = 0;
  virtual void PerformIteration() = 0;
  virtual void Release() = 0;

  /// \brief Copies current primal dual solution pair (x,y) to the host.
  virtual void current_solution(std::vector<T>& primal_sol, std::vector<T>& dual_sol) const = 0;

  /// \brief Returns norm of the primal residual |Ax - z|.
  virtual T primal_residual() const = 0;

  /// \brief Returns norm of the dual residual |A^T y + w|.
  virtual T dual_residual() const = 0;

  /// \brief Returns norm of the primal variable "z", used for stopping criterion.
  virtual T primal_var_norm() const = 0;

  /// \brief Returns norm of the dual variable "w", used for stopping criterion.
  virtual T dual_var_norm() const = 0;

  // returns amount of gpu memory required in bytes
  virtual size_t gpu_mem_amount() const = 0;

protected:
  std::shared_ptr<Problem<T> > problem_;
};

#endif
