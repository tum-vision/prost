#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include <memory>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

template<typename T> class Prox;
template<typename T> class Block;
template<typename T> class LinearOperator;

/**
 * @brief Contains all information describing the graph form problem
 *
 *          min_x,y   f(D^{-1} y) + g(E x)  s.t. y = DAE x. 
 *
 */

template<typename T>
class Problem 
{
public:
  enum Scaling
  {
    kScalingIdentity,
    kScalingAlpha,
    kScalingCustom,
  };

  typedef std::vector<std::shared_ptr<Prox<T> > > ProxList;

  Problem();
  virtual ~Problem();

  void AddBlock(std::shared_ptr<Block<T> > block);
  void AddProx_g(std::shared_ptr<Prox<T> > prox);
  void AddProx_f(std::shared_ptr<Prox<T> > prox);
  void AddProx_gstar(std::shared_ptr<Prox<T> > prox);
  void AddProx_fstar(std::shared_ptr<Prox<T> > prox);

  // builds the linear operator and checks if prox cover the 
  // whole domain
  void Initialize();
  void Release();

  // sets a predefined problem scaling, has to be called after Init
  void SetScalingCustom(
    const std::vector<T>& left, 
    const std::vector<T>& right);

  // computes a scaling using the Diagonal Preconditioners
  // proposed in Pock, Chambolle ICCV '11. Has to be called after Init.
  void SetScalingAlpha(T alpha);

  //
  void SetScalingIdentity();

  std::shared_ptr<LinearOperator<T> > linop() const { return linop_; }
  thrust::device_vector<T>& scaling_left() { return scaling_left_; }
  thrust::device_vector<T>& scaling_right() { return scaling_right_; }
  const ProxList& prox_f() const { return prox_f_; }
  const ProxList& prox_g() const { return prox_g_; }
  const ProxList& prox_fstar() const { return prox_fstar_; }
  const ProxList& prox_gstar() const { return prox_gstar_; }

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  size_t gpu_mem_amount() const;

  /// \brief Estimates the norm of the scaled linear operator via power iteration.
  T normest(T tol = 1e-6, int max_iters = 100);
  
protected:
  size_t nrows_, ncols_; // problem size

  // matrix A
  std::shared_ptr<LinearOperator<T> > linop_;

  // left/right preconditioners (D and E)
  typename Problem<T>::Scaling scaling_type_;
  thrust::device_vector<T> scaling_left_; 
  thrust::device_vector<T> scaling_right_;
  std::vector<T> scaling_left_custom_;
  std::vector<T> scaling_right_custom_;
  T scaling_alpha_;

  // proximal operators (overcomplete)
  ProxList prox_f_;
  ProxList prox_g_;
  ProxList prox_fstar_;
  ProxList prox_gstar_;
};

#endif
