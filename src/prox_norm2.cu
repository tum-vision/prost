#include "prox_norm2.hpp"
#include "util/cuwrap.hpp"

template<class ProxFunc1D, typename real>
__global__
void ProxNorm2Kernel(
    const real *d_arg,
    real *d_result,
    real tau,
    const real *d_tau,
    const ProxFunc1D prox,
    const real *d_a,
    const real *d_b,
    const real *d_c,
    const real *d_d,
    const real *d_e,
    real cf_a,
    real cf_b,
    real cf_c,
    real cf_d,
    real cf_e,
    int index,
    int count,
    int dim,
    bool interleaved,
    bool invert_steps) {
  
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx >= count)
    return;
  
  // compute dim-dimensional 2-norm at each point
  real norm = 0;
  int global_idx;
    
  for(int i = 0; i < dim; i++)
  {
    if(interleaved)
      global_idx = index + th_idx * dim + i;
    else
      global_idx = index + th_idx + count * i;

    real arg = d_arg[global_idx];
    norm += arg * arg;
  }
    
  if(norm > 0)
  {
    norm = cuwrap::sqrt<real>(norm); 

    // 1d prox, again this isn't too pretty
    if(d_a != 0) cf_a = d_a[th_idx];
    if(d_b != 0) cf_b = d_b[th_idx];
    if(d_c != 0) cf_c = d_c[th_idx];
    if(d_d != 0) cf_d = d_d[th_idx];
    if(d_e != 0) cf_e = d_e[th_idx];

    // handle diagonal preconditioner
    if(d_tau != 0) {
      if(interleaved)
        tau *= d_tau[index + th_idx * dim];
      else
        tau *= d_tau[index + th_idx];
    }

    if(invert_steps)
      tau = 1. / tau;

    // do scaled 1d prox
    const real arg = cf_a * (norm * tau - cf_d) / (cf_e + tau) - cf_b;
    const real step = (cf_e + tau) / (cf_c * cf_a * cf_a);
    const real prox_result = (prox.Apply(arg, step) + cf_b) / cf_a;

    // combine together for result
    for(int i = 0; i < dim; i++)
    {
      if(interleaved)
        global_idx = index + th_idx * dim + i;
      else
        global_idx = index + th_idx + count * i;

      d_result[global_idx] = prox_result * d_arg[global_idx] / norm;
    }
  }
  else // in that case, the result is zero.
  {
    for(int i = 0; i < dim; i++)
    {
      if(interleaved)
        global_idx = index + th_idx * dim + i;
      else
        global_idx = index + th_idx + count * i;

      d_result[global_idx] = 0;
    }
  }
}

ProxNorm2::ProxNorm2(
    int index,
    int count,
    int dim,
    bool interleaved,
    const Prox1DCoefficients& coeffs,
    const Prox1DFunction& func)
    
    : Prox1D(index, count, coeffs, func)
{  
  dim_ = dim;
  interleaved_ = interleaved;
  diagsteps_ = false;
}

ProxNorm2::~ProxNorm2() {
}

void ProxNorm2::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count_ + block.x - 1) / block.x, 1, 1);

  #define CALL_PROX_NORM2_KERNEL(Func) \
    ProxNorm2Kernel<Func<real>, real> \
        <<<grid, block>>>( \
            d_arg, \
            d_result, \
            tau, \
            d_tau, \
            Func<real>(), \
            d_coeffs_[0], d_coeffs_[1], d_coeffs_[2], d_coeffs_[3], d_coeffs_[4], \
            coeffs_.a[0], coeffs_.b[1], coeffs_.c[2], coeffs_.d[3], coeffs_.e[4], \
            index_, count_, dim_, interleaved_, invert_step)
  
  switch(func_) {
    case kZero:
      CALL_PROX_NORM2_KERNEL(Prox1DZero);
      break;

    case kAbs:
      CALL_PROX_NORM2_KERNEL(Prox1DAbs);
      break;

    case kSquare:
      CALL_PROX_NORM2_KERNEL(Prox1DSquare);
      break;

    case kIndicatorLeq:
      CALL_PROX_NORM2_KERNEL(Prox1DIndicatorLeq);
      break;

    case kIndicatorEq:
      CALL_PROX_NORM2_KERNEL(Prox1DIndicatorEq);
      break;

    case kIndicatorAbsLeq:
      CALL_PROX_NORM2_KERNEL(Prox1DIndicatorAbsLeq);
      break;

    default:
      break;
  }

  cudaDeviceSynchronize();
}
