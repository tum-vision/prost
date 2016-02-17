#include "prost/prox/prox_transform.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
__global__ 
void ProxTransformPrescaleArgument(
  T *scaled_arg,
  const T *arg,
  const T *tau_diag,
  const T *dev_a,
  const T *dev_b,
  const T *dev_d,
  const T *dev_e,
  T a, T b, T d, T e, T tau, size_t n)
{
  size_t tx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tx >= n)
    return;

  a = (nullptr == dev_a) ? a : dev_a[tx];
  b = (nullptr == dev_b) ? b : dev_b[tx];
  d = (nullptr == dev_d) ? d : dev_d[tx];
  e = (nullptr == dev_e) ? e : dev_e[tx];

  scaled_arg[tx] = (a * (arg[tx] - tau * tau_diag[tx] * d)) / (1 + tau * tau_diag[tx] * e) - b;
}

template<typename T>
__global__
void ProxTransformPrescaleStepSize(
  T *scaled_tau,
  const T *tau_diag,
  const T *dev_a,
  const T *dev_c,
  const T *dev_e,
  T a, T c, T e, T tau, size_t n)
{
  size_t tx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tx >= n)
    return;

  a = (nullptr == dev_a) ? a : dev_a[tx];
  c = (nullptr == dev_c) ? c : dev_c[tx];
  e = (nullptr == dev_e) ? e : dev_e[tx];

  scaled_tau[tx] = (a * a * c * tau * tau_diag[tx]) / (1 + tau * tau_diag[tx] * e);
}

template<typename T>
__global__
void ProxTransformPostscale(
  T *result,
  const T *dev_a,
  const T *dev_b,
  T a, T b, size_t n)
{
  size_t tx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tx >= n)
    return;

  a = (nullptr == dev_a) ? a : dev_a[tx];
  b = (nullptr == dev_b) ? b : dev_b[tx];

  result[tx] = (result[tx] + b) / a;
}

template<typename T>
ProxTransform<T>::ProxTransform(
    shared_ptr<Prox<T> > inner_fn,
    const vector<T>& a,
    const vector<T>& b,
    const vector<T>& c,
    const vector<T>& d,
    const vector<T>& e)
  : Prox<T>(*inner_fn), host_a_(a), host_b_(b), host_c_(c), host_d_(d), host_e_(e), inner_fn_(inner_fn)
{
}

template<typename T>
void ProxTransform<T>::Initialize() 
{
  for(T& a : host_a_)
    if(a == 0)
      throw Exception("ProxTransform: Vector 'a' isn't allowed to contain zero element. (Division by zero)");

  // TODO check for allowed dimensions and throw exceptions if necessary

  try 
  {
    if(host_a_.size() > 1) dev_a_ = host_a_;
    if(host_b_.size() > 1) dev_b_ = host_b_;
    if(host_c_.size() > 1) dev_c_ = host_c_;
    if(host_d_.size() > 1) dev_d_ = host_d_;
    if(host_e_.size() > 1) dev_e_ = host_e_;

    scaled_arg_.resize(this->size_);
    scaled_tau_.resize(this->size_);
  } 
  catch(const std::bad_alloc &e)
  {
    std::stringstream ss;
	ss << "Out of memory: " << e.what();
    throw Exception(ss.str());
  }

  inner_fn_->Initialize();
}

template<typename T>
void ProxTransform<T>::Release() 
{
  inner_fn_->Release();
}

template<typename T>
size_t ProxTransform<T>::gpu_mem_amount() const
{
  return 
     ((host_a_.size() * (host_a_.size() > 1)) + 
      (host_b_.size() * (host_b_.size() > 1)) +
      (host_c_.size() * (host_c_.size() > 1)) +
      (host_d_.size() * (host_d_.size() > 1)) +
      (host_e_.size() * (host_e_.size() > 1))) * sizeof(T) + 
      inner_fn_->gpu_mem_amount() +
      2 * this->size_ * sizeof(T);
}

template<typename T>
void ProxTransform<T>::get_separable_structure(vector<std::tuple<size_t, size_t, size_t> >& sep)
{
  inner_fn_->get_separable_structure(sep);
}

template<typename T>
void ProxTransform<T>::EvalLocal(
    const typename device_vector<T>::iterator& result_beg,
    const typename device_vector<T>::iterator& result_end,
    const typename device_vector<T>::const_iterator& arg_beg,
    const typename device_vector<T>::const_iterator& arg_end,
    const typename device_vector<T>::const_iterator& tau_beg,
    const typename device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->size_ + block.x - 1) / block.x, 1, 1);

  // scale argument and step size
  ProxTransformPrescaleArgument<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(scaled_arg_.data()),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      (host_a_.size() > 1) ? thrust::raw_pointer_cast(dev_a_.data()) : nullptr,
      (host_b_.size() > 1) ? thrust::raw_pointer_cast(dev_b_.data()) : nullptr,
      (host_d_.size() > 1) ? thrust::raw_pointer_cast(dev_d_.data()) : nullptr,
      (host_e_.size() > 1) ? thrust::raw_pointer_cast(dev_e_.data()) : nullptr,
      host_a_[0], host_b_[0], host_d_[0], host_e_[0], tau, this->size_);

  ProxTransformPrescaleStepSize<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(scaled_tau_.data()),
      thrust::raw_pointer_cast(&(*tau_beg)),
      (host_a_.size() > 1) ? thrust::raw_pointer_cast(dev_a_.data()) : nullptr,
      (host_c_.size() > 1) ? thrust::raw_pointer_cast(dev_c_.data()) : nullptr,
      (host_e_.size() > 1) ? thrust::raw_pointer_cast(dev_e_.data()) : nullptr,
      host_a_[0], host_c_[0], host_e_[0], tau, this->size_);

  // compute prox on scaled argument
  inner_fn_->EvalLocal(
    result_beg,
    result_end,
    scaled_arg_.begin(),
    scaled_arg_.end(),
    scaled_tau_.begin(),
    scaled_tau_.end(),
    1,
    invert_tau);

  // rescale result
  ProxTransformPostscale<T>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      (host_a_.size() > 1) ? thrust::raw_pointer_cast(dev_a_.data()) : nullptr,
      (host_b_.size() > 1) ? thrust::raw_pointer_cast(dev_b_.data()) : nullptr,
      host_a_[0], host_b_[0], this->size_);
}

template class ProxTransform<float>;
template class ProxTransform<double>;

} // namespace prost
