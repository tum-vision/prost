#include "prox/prox_norm2.hpp"

#include "config.hpp"
#include "util/cuwrap.hpp"

#include <iostream>

template<typename T, class ELEM_OPERATION>
__global__
void ProxSeparableSumKernel(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau,
    bool interleaved,
    ELEM_OPERATION::Data* d_data,
    size_t count)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    ELEM_OPERATION::Data data = d_data[tx];
    
    Vector(this, d_res, tx) d_res_v;
    Vector(this, d_arg, tx) d_arg_v;
    Vector(this, d_tau, tx) d_tau_v;
    

    ELEM_OPERATION(d_res_v, d_arg_v, d_tau_v, tau, invert_tau, data);
  }
}

template<typename T>
ProxNorm2<T>::ProxNorm2(size_t index,
                        size_t count,
                        const Prox1DCoefficients<T>& coeffs,
                        const Prox1DFunction& func) :
    Prox1D<T>(index, count, coeffs, func)
{
  this->diagsteps_ = false;
}

template<typename T>
ProxNorm2<T>::~ProxNorm2() {
}

template<typename T>
void ProxNorm2<T>::EvalLocal(T *d_arg,
                             T *d_res,
                             T *d_tau,
                             T tau,
                             bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  #define CALL_PROX_NORM2_KERNEL(Func) \
    ProxNorm2Kernel<Func<T>, T> \
        <<<grid, block>>>( \
            d_arg, \
            d_res, \
            d_tau, \
            tau, \
            Func<T>(), \
            this->coeffs_dev_, \
            this->count_, this->dim_, this->interleaved_, invert_tau)
  
  switch(this->func_) {
    case kZero:
      CALL_PROX_NORM2_KERNEL(Prox1DZero);
      break;

    case kAbs:
      CALL_PROX_NORM2_KERNEL(Prox1DAbs);
      break;

    case kSquare:
      CALL_PROX_NORM2_KERNEL(Prox1DSquare);
      break;

    case kMaxPos0:
      CALL_PROX_NORM2_KERNEL(Prox1DMaxPos0);
      break;

    case kIndLeq0:
      CALL_PROX_NORM2_KERNEL(Prox1DIndLeq0);
      break;

    case kIndEq0:
      CALL_PROX_NORM2_KERNEL(Prox1DIndEq0);
      break;

    case kIndBox01:
      CALL_PROX_NORM2_KERNEL(Prox1DIndBox01);
      break;

    case kL0:
      CALL_PROX_NORM2_KERNEL(Prox1DL0);
      break;

    case kHuber:
      CALL_PROX_NORM2_KERNEL(ProxHuber);
      break;

    default:
      break;
  }
}

// Explicit template instantiation
template class ProxNorm2<float>;
template class ProxNorm2<double>;
