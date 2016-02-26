#include <iostream>
#include <sstream>

#include "prost/prox/prox_ind_epi_quad.hpp"
#include "prost/prox/vector.hpp"
#include "prost/prox/helper.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
struct Coefficients {
  const T* dev_a;
  const T* dev_b;
  const T* dev_c;


  T a;
  T c;
};

template<typename T>
__global__
void ProxIndEpiQuadKernel(
  T *d_res,
  const T *d_arg,
  size_t count,
  size_t dim,
  Coefficients<T> coeffs)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> x(count, dim-1, false, tx, d_res);
    const Vector<const T> x0(count, dim-1, false, tx, d_arg);
    T& y = d_res[count * (dim-1) + tx];
    const T y0 = d_arg[count * (dim-1) + tx];

    const T a = coeffs.dev_a == nullptr ? coeffs.a : coeffs.dev_a[tx];
    const Vector<const T> b(count, dim-1, false, tx, coeffs.dev_b);
    const T c = coeffs.dev_c == nullptr ? coeffs.c : coeffs.dev_c[tx];

    T sq_norm_b = static_cast<T>(0);
    for(size_t i = 0; i < dim-1; i++) {
      T val = b[i];
      x[i] = x0[i] + val / a;
      sq_norm_b += val * val;
    }
    

    helper::ProjectEpiQuadNd<T>(x, y0 / a + (0.5 / (a*a)) * sq_norm_b - c / a, 0.5, x, y, dim-1);
      
    for(size_t i = 0; i < dim-1; i++) {
      x[i] -= b[i] / a;
    }

    y = y * a - (0.5 / a) * sq_norm_b + c;
  }
}


template<typename T>
void 
ProxIndEpiQuad<T>::EvalLocal(
  const typename thrust::device_vector<T>::iterator& result_beg,
  const typename thrust::device_vector<T>::iterator& result_end,
  const typename thrust::device_vector<T>::const_iterator& arg_beg,
  const typename thrust::device_vector<T>::const_iterator& arg_end,
  const typename thrust::device_vector<T>::const_iterator& tau_beg,
  const typename thrust::device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  Coefficients<T> coeffs;
  if(a_.size() != 1) {
    coeffs.dev_a = thrust::raw_pointer_cast(&(d_a_[0]));
  } else {
    coeffs.dev_a = nullptr;
    coeffs.a = a_[0];
  }

  coeffs.dev_b = thrust::raw_pointer_cast(&(d_b_[0]));


  if(c_.size() != 1) {
    coeffs.dev_c = thrust::raw_pointer_cast(&(d_c_[0]));
  } else {
    coeffs.dev_c = nullptr;
    coeffs.c = c_[0];
  }

  ProxIndEpiQuadKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      this->count_,
      this->dim_,
      coeffs);
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

template<typename T>
void
ProxIndEpiQuad<T>::Initialize() 
{
    if(a_.size() != this->count_ && a_.size() != 1)
      throw Exception("Wrong input: Coefficient a has to have dimension count or 1!");

    for(T& a : a_) {
      if(a <= 0)
        throw Exception("Wrong input: Coefficient a must be greater 0!");
    }

    if(b_.size() != this->count_*(this->dim_-1) && b_.size() != this->dim_-1)
      throw Exception("Wrong input: Coefficient b has to have dimension count*(dim-1) or dim-1!");

    if(c_.size() != this->count_ && c_.size() != 1)
      throw Exception("Wrong input: Coefficient c has to have dimension count or 1!");

    try
    {
      d_a_ = a_;
      d_b_ = b_;
      d_c_ = c_;
    }
    catch(std::bad_alloc &e)
    {
      throw Exception(e.what());
    }
    catch(thrust::system_error &e)
    {
      throw Exception(e.what());
    }
    
}

// Explicit template instantiation
template class ProxIndEpiQuad<float>;
template class ProxIndEpiQuad<double>;

}
