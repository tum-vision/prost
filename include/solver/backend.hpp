#ifndef BACKEND_HPP_
#define BACKEND_HPP_

class Problem;

/**
 * @brief Algorithm implementation for solving graph form problems.
 *
 */
template<typename T>
class Backend {
public:
  Backend(std::unique_ptr<Problem<T> > problem);
  virtual ~Backend();

  virtual void Initialize() = 0;
  virtual void PerformIteration() = 0;
  virtual void Release() = 0;

  virtual void current_solution(std::vector<T>& primal, std::vector<T>& dual) const = 0;
  virtual T primal_residual() const = 0;
  virtual T dual_residual() const = 0;

  // returns amount of gpu memory required in bytes
  virtual size_t gpu_mem_amount() const = 0;

protected:
  std::weak_ptr<Problem<T> > problem_;
};

#endif
