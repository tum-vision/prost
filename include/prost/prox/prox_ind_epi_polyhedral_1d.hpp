#ifndef PROST_PROX_IND_EPI_POLYHEDRAL_1D_HPP_
#define PROST_PROX_IND_EPI_POLYHEDRAL_1D_HPP_

#include "prost/prox/prox_separable_sum.hpp"

namespace prost {

/// 
/// \brief Computes orthogonal projection onto the epigraph
///        of a one-dimensional _convex_ piecewise linear function.
/// 
///        The piecewise linear function is described by a set of points
///        pt_x, pt_y, beginning slope alpha and ending slope beta. The
///        variable count_vec indicates the number of points at the pixel
///        and index_vec specifies the pointer into pt_x, pt_y.
///
///        See also right part of Fig. 4 in the paper:
///        http://arxiv.org/abs/1512.01383
///
template<typename T>
class ProxIndEpiPolyhedral1D : public ProxSeparableSum<T> {
public:
  ProxIndEpiPolyhedral1D(
    size_t index,
    size_t count,
    bool interleaved,
    const vector<T>& pt_x, 
    const vector<T>& pt_y,
    const vector<T>& alpha,
    const vector<T>& beta,
    const vector<size_t>& count_vec,
    const vector<size_t>& index_vec);

  virtual ~ProxIndEpiPolyhedral1D() { }

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
  vector<T> host_pt_x_, host_pt_y_, host_alpha_, host_beta_;
  vector<size_t> host_count_, host_index_;

  device_vector<T> dev_pt_x_, dev_pt_y_, dev_alpha_, dev_beta_;
  device_vector<size_t> dev_count_, dev_index_;
};

} // namespace prost

#endif // PROST_PROX_IND_EPI_POLYHEDRAL_1D_HPP_
