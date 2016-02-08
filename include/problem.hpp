#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include <memory>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

template<typename T> class Prox;
template<typename T> class Block;
template<typename T> class LinearOperator;

/// @brief Contains all information describing the graph form problem
/// 
///         min_{x,z} g(\Tau^{1/2} x) + f(\Sigma^{-1/2} z) 
/// 
///         s.t. z = \Sigma^{1/2} K \Tau^{1/2} x. 
/// 
///         scaling_left_host_ = Sigma,
///         scaling_right_host_ = Tau.
/// 
template<typename T>
class Problem 
{
public:
  enum Scaling
  {
    kScalingIdentity,
    kScalingAlpha, // Pock, Chambolle ICCV '11 preconditioning
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

  /// \brief Builds the linear operator and checks if prox cover the 
  ///        whole domain.
  void Initialize();
  void Release();

  /// \brief Sets a predefined problem scaling.
  /// \param[in] left Left scaling/preconditioner diagonal matrix Sigma^{1/2}.
  /// \param[in] right Right scaling/preconditioner diagonal matrix Tau^{1/2}.
  void SetScalingCustom(
    const std::vector<T>& left, 
    const std::vector<T>& right);

  /// \brief Set the scaling to the Diagonal Preconditioners
  ///        proposed in Pock, Chambolle ICCV '11. 
  void SetScalingAlpha(T alpha);

  /// \brief Set the preconditioners to Sigma = Tau = Identity.
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
  T normest(T tol = 1e-6, int max_iters = 250);
  
protected:
  size_t nrows_, ncols_; // problem size

  /// \brief matrix K
  std::shared_ptr<LinearOperator<T> > linop_;

  typename Problem<T>::Scaling scaling_type_;

  /// \brief (squared) Left-preconditioner Sigma
  thrust::device_vector<T> scaling_left_; 

  /// \brief (squared) Right-preconditioner Tau
  thrust::device_vector<T> scaling_right_;

  /// \brief User-defined (squared) left-preconditioner Sigma
  std::vector<T> scaling_left_host_;

  /// \brief User-defined (squared) right-preconditioner Tau
  std::vector<T> scaling_right_host_;

  /// \brief alpha for Pock-preconditioning
  T scaling_alpha_;

  /// \brief proximal operators (may be overcomplete).
  ProxList prox_f_;
  ProxList prox_g_;
  ProxList prox_fstar_;
  ProxList prox_gstar_;

private:
  /// \brief Averages the values for the preconditioner at the entries where
  ///        prox does not allow diagonal step sizes.
  void AveragePreconditioners(
    std::vector<T>& precond,
    const ProxList& prox);
};

#endif
