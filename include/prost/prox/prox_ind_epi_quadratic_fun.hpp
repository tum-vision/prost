#ifndef PROST_PROX_IND_EPI_QUADRATIC_FUN_HPP_
#define PROST_PROX_IND_EPI_QUADRATIC_FUN_HPP_

#include <array>
#include <vector>
#include <thrust/device_vector.h>

#include "prost/prox/prox_separable_sum.hpp"
#include "prost/prox/vector.hpp"
#include "prost/common.hpp"

namespace prost {

///
/// \brief Implements projection onto epigraph of quadratic function s.t. a/2 x^T x + b^T x + c >= y
///
template<typename T>
class ProxIndEpiQuadraticFun : public ProxSeparableSum<T> 
{
public:    
  ProxIndEpiQuadraticFun(
      size_t index, 
      size_t count, 
      size_t dim, 
      bool interleaved, 
      bool diagsteps, 
      std::vector<T> a,
      std::vector<T> b,
      std::vector<T> c)
      : ProxSeparableSum<T>(index, count, dim, interleaved, diagsteps), a_(a), b_(b), c_(c) { }

  virtual void Initialize();
  
  virtual size_t gpu_mem_amount() const
  {
    return (this->count_*this->dim_ + this->count_) *sizeof(T);
  }

  #ifdef __CUDACC__

  ///
  /// \brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
  ///        the parabola y >= \alpha ||x||^2 with \alpha > 0.
  ///
  /// TODO: Move this to math_functions.hpp?!
  /// 
  inline __device__ 
  static void ProjectSimple(
    const Vector<T>& x0, const T y0, const T alpha, Vector<T>& x, T& y, size_t dim)
  {

  T sq_norm_x0 = static_cast<T>(0);
  for(size_t i = 0; i < dim; i++) {
    sq_norm_x0 += x0[i] * x0[i];
  }
  const T norm_x0 = sqrt(sq_norm_x0);
  
  // nothing to do?
  if(y0 >= alpha * sq_norm_x0) {
    for(size_t i = 0; i < dim; i++) 
      x[i] = x0[i];

    y = y0;
  }
  else {
    const T a = 2. * alpha * norm_x0;
    const T b = 2. * (1. - 2. * alpha * y0) / 3.;
    T d, v;

    if(b < 0) {
      const T sq = pow(-b, static_cast<T>(3. / 2.));
      d = (a - sq) * (a + sq);      
    }
    else {
      d = a * a + b * b * b;
    }

    if(d >= 0) {
      const T c = pow(a + sqrt(d), static_cast<T>(1. / 3.));

      if(abs(c) > 1e-6)
        v = c - b / c;
      else
        v = 0;
    }
    else {
      v = 2 * sqrt(-b) * cos(acos(a / pow(-b, static_cast<T>(3. / 2.))) / static_cast<T>(3.));
    }

    if(norm_x0 > 0) { 
      for(size_t i = 0; i < dim; i++) {
        x[i] = (v / (2. * alpha)) * (x0[i] / norm_x0);
      }
    }
    else {
      for(size_t i = 0; i < dim; i++) {
        x[i] = 0;
      }
    }

    T sq_norm_x = static_cast<T>(0);
    for(size_t i = 0; i < dim; i++) {
      sq_norm_x += x[i] * x[i];
    }

    y = alpha * sq_norm_x;
   
  }
  }
  #endif

   
protected:
  virtual void EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
  
private:    
  thrust::device_vector<T> d_a_;
  thrust::device_vector<T> d_b_;
  thrust::device_vector<T> d_c_;  
 
  std::vector<T> a_;
  std::vector<T> b_;
  std::vector<T> c_;
};

} // namespace prost

#endif // PROST_PROJ_EPI_QUADRATIC_FUN_HPP_
