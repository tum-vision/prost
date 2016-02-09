#include <iostream>
#include <sstream>

#include "prost/prox/proj_epi_quadratic_fun.hpp"
#include "prost/prox/vector.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
__global__
void ProjEpiQuadraticFunKernel(
  T *d_res,
  const T *d_arg,
  const T *d_tau,
  T tau,
  bool invert_tau,
  size_t count,
  size_t dim,
  const T* d_a,
  const T* d_b,
  const T* d_c,
  bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> res(count, dim, interleaved, tx, d_res);
    Vector<T> arg(count, dim, interleaved, tx, d_arg);
    Vector<T> tau_diag(count, dim, interleaved, tx, d_tau);
    const T a = d_a[tx];
    Vector<T> b(count, dim-1, interleaved, tx, d_b);
    const T c = d_c[tx];

    const T alpha = 0.5 * a;
    ProjectSimple<T>(arg, arg[dim-1], alpha, res, res[dim-1], dim-1);
      
    T sq_norm_b = static_cast<T>(0);
    for(size_t i = 0; i < dim-1; i++) {
      T val = b[i];
      res[i] -= val;
      sq_norm_b += val * val;
    }

    res[dim-1] = res[dim-1] + a * c - 0.5 * a * a * sq_norm_b;
  }
}


template<typename T>
void 
ProjEpiQuadraticFun<T>::EvalLocal(
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

  ProjEpiQuadraticFunKernel<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      tau,
      invert_tau,
      this->count_,
      this->dim_,
      thrust::raw_pointer_cast(&(d_a_[0])),
      thrust::raw_pointer_cast(&(d_b_[0])),
      thrust::raw_pointer_cast(&(d_c_[0])),
      this->interleaved_);
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
ProjEpiQuadraticFun<T>::Initialize() 
{
    if(a_.size() != this->count_)
      throw Exception("Wrong input: Coefficient a has to have dimension count!");

    if(b_.size() != this->count_*(this->dim_-1))
      throw Exception("Wrong input: Coefficient b has to have dimension count*dim!");

    if(c_.size() != this->count_)
      throw Exception("Wrong input: Coefficient c has to have dimension count!");

    try
    {
      d_a_.resize(this->count_);
      thrust::copy(b_.begin(), b_.end(), d_b_.begin());

      d_b_.resize(this->count_*(this->dim_-1));
      thrust::copy(b_.begin(), b_.end(), d_b_.begin());

      d_c_.resize(this->count_);
      thrust::copy(c_.begin(), c_.end(), d_c_.begin());
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
template class ProjEpiQuadraticFun<float>;
template class ProjEpiQuadraticFun<double>;

}