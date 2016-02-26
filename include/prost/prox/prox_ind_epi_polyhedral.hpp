#ifndef PROST_PROX_IND_EPI_POLYHEDRAL_HPP_
#define PROST_PROX_IND_EPI_POLYHEDRAL_HPP_

#include "prost/common.hpp"
#include "prost/prox/prox_separable_sum.hpp"

namespace prost {

///
/// \brief Computes the orthogonal projection onto the 
///        epigraph of a piecewise linear function f described
///        by a maximum over linear functions
/// 
///        f(x) = max_{i=1..m} <a_i, x> + b_i
/// 
///        Here, the a_i are dim_ - 1 dimensional and the b_i are 1 
///        dimensional coefficients.
///
///        The prox solves the following optimization problem:
/// 
///        min_{x,t} (x - y)^2 + (t - u)^2
/// 
///        s.t. t >= <a_i, x> + b_i, i = 1..m
/// 
template<typename T>
class ProxIndEpiPolyhedral : public ProxSeparableSum<T> {
public:
  /// \brief Constructor.
  /// 
  /// \param index     Start of the prox.
  /// \param count     Number of elements in the sum (e.g., number of pixels)
  /// \param dim       Dimension of the prox. f(x) is dim-1 dimensional.
  /// \param coeffs_a  A list of the m coefficients a_i, for every function, stored linearly.
  /// \param coeffs_b  A list of the m coefficients b_i, for every function, stored linearly.
  /// \param count_vec A list containing the number m for every function.
  /// \param index_vec A list containing the index into coeffs_a and coeffs_b for every function. 
  ///                  This is equal to the running sum over count_vec.
  ProxIndEpiPolyhedral(
    size_t index,
    size_t count,
    size_t dim, 
    const vector<T>& coeffs_a,
    const vector<T>& coeffs_b, 
    const vector<size_t>& count_vec,
    const vector<size_t>& index_vec);

  virtual ~ProxIndEpiPolyhedral() {}

  virtual void Initialize();
  virtual size_t gpu_mem_amount() const;

protected:
  virtual void EvalLocal(
    const typename device_vector<T>::iterator& result_beg,
    const typename device_vector<T>::iterator& result_end,
    const typename device_vector<T>::const_iterator& arg_beg,
    const typename device_vector<T>::const_iterator& arg_end,
    const typename device_vector<T>::const_iterator& tau_beg,
    const typename device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);

private:
  vector<T> host_coeffs_a_, host_coeffs_b_;
  vector<size_t> host_count_, host_index_;

  device_vector<T> dev_coeffs_a_, dev_coeffs_b_;
  device_vector<size_t> dev_count_, dev_index_;
};

} // namespace prost

#endif // PROST_PROX_IND_EPI_POLYHEDRAL_HPP_
