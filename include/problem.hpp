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
  typedef std::vector<std::shared_ptr<Prox<T> > > ProxList;

  Problem();
  virtual ~Problem();

  void AddBlock(Block<T> *block);
  void AddProx_g(Prox<T> *prox);
  void AddProx_f(Prox<T> *prox);
  void AddProx_gstar(Prox<T> *prox);
  void AddProx_fstar(Prox<T> *prox);

  // builds the linear operator and checks if prox cover the 
  // whole domain
  void Init();
  void Release();

  // sets a predefined problem scaling, has to be called after Init
  void SetScalingPredefined(
    const std::vector<T>& left, 
    const std::vector<T>& right);

  // computes a scaling using the Diagonal Preconditioners
  // proposed in Pock, Chambolle ICCV '11. Has to be called after Init.
  void SetScalingAlphaPrecond(T alpha);

  //
  void SetScalingIdentity();

  std::weak_ptr<LinearOperator<T> > linop() const { return std::weak_ptr<LinearOperator<T> >(linop_); }
  thrust::device_ptr<const T> scaling_left() const { return &scaling_left_[0]; }
  thrust::device_ptr<const T> scaling_right() const { return &scaling_right_[0]; }
  const ProxList& prox_f() const { return prox_f_; }
  const ProxList& prox_g() const { return prox_g_; }
  const ProxList& prox_fstar() const { return prox_fstar_; }
  const ProxList& prox_gstar() const { return prox_gstar_; }

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

protected:
  size_t nrows_, ncols_; // problem size

  // matrix A
  std::shared_ptr<LinearOperator<T> > linop_;

  // left/right preconditioners (D and E)
  thrust::device_vector<T> scaling_left_; 
  thrust::device_vector<T> scaling_right_; 

  // proximal operators (overcomplete)
  ProxList prox_f_;
  ProxList prox_g_;
  ProxList prox_fstar_;
  ProxList prox_gstar_;
};

#endif
