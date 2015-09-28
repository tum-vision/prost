#include "prox/prox_norm2.hpp"

#include "config.hpp"
#include "util/cuwrap.hpp"

template<class ProxFunc1D, typename T>
__global__
void ProxNorm2Kernel(
    T *d_arg,
    T *d_res,
    T* d_tau,
    T tau,
    const ProxFunc1D prox,
    Prox1DCoeffsDevice<T> coeffs,
    size_t count,
    size_t dim,
    bool interleaved,
    bool invert_tau)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    // compute dim-dimensional 2-norm at each point
    T norm = 0;
    size_t index;

    for(size_t i = 0; i < dim; i++) {
      index = interleaved ? (tx * dim + i) : (tx + count * i);
      const T arg = d_arg[index];
      norm += arg * arg;
    }
    
    if(norm > 0) {
      norm = sqrt(norm);

      // read value for vector coefficients
      for(size_t i = 0; i < PROX_1D_NUM_COEFFS; i++) {
        if(coeffs.d_ptr[i] != NULL)
          coeffs.val[i] = coeffs.d_ptr[i][tx];
      }

      // compute step-size
      index = interleaved ? (tx * dim) : tx;
      tau = invert_tau ? (1. / (tau * d_tau[index])) : (tau * d_tau[index]);

      // compute scaled prox argument and step 
      const T arg = ((coeffs.val[0] * (norm - coeffs.val[3] * tau)) /
                     (1. + tau * coeffs.val[4])) - coeffs.val[1];
    
      const T step = (coeffs.val[2] * coeffs.val[0] * coeffs.val[0] * tau) /
                     (1. + tau * coeffs.val[4]);
      
      // compute prox
      const T prox_result = (prox.Eval(arg, step, coeffs.val[5], coeffs.val[6]) +
                             coeffs.val[1]) / coeffs.val[0];

      // combine together for result
      for(size_t i = 0; i < dim; i++) {
        index = interleaved ? (tx * dim + i) : (tx + count * i);
        d_res[index] = prox_result * d_arg[index] / norm;
      }
    }
    else { // in that case, the result is zero. 
      for(size_t i = 0; i < dim; i++) {
        index = interleaved ? (tx * dim + i) : (tx + count * i);
        d_res[index] = 0;
      }
    }
  }
}

template<typename T>
ProxNorm2<T>::ProxNorm2(size_t index,
                        size_t count,
                        size_t dim,
                        bool interleaved,
                        const Prox1DCoefficients<T>& coeffs,
                        const Prox1DFunction& func) :
    Prox1D<T>(index, count, coeffs, func)
{
  this->dim_ = dim;
  this->interleaved_ = interleaved;
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

    default:
      break;
  }
}

// Explicit template instantiation
template class ProxNorm2<float>;
template class ProxNorm2<double>;
