#include "prox_1d.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

// TODO: should this be handled inside bind_matlab code?
Prox1DFunction Prox1DFunctionFromString(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  static std::string names[] = {
    "zero",
    "abs",
    "square",
    "indicator_leq",
    "indicator_eq",
    "indicator_abs_leq" };

  static Prox1DFunction funcs[] = {
    kZero,
    kAbs,
    kSquare,
    kIndicatorLeq,
    kIndicatorEq,
    kIndicatorAbsLeq };

  for(int i = 0; i < kNumProx1DFunctions; i++)
    if(names[i] == name)
      return funcs[i];

  return kInvalidProx;
}

template<class ProxFunc1D, typename real>
__global__
void Prox1DKernel(
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
    bool invert_step)
{
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx < count) {
    int global_idx = index + th_idx;

    // this doesn't look too good, but no idea how to it cleaner :-)
    if(d_a != 0) cf_a = d_a[th_idx];
    if(d_b != 0) cf_b = d_b[th_idx];
    if(d_c != 0) cf_c = d_c[th_idx];
    if(d_d != 0) cf_d = d_d[th_idx];
    if(d_e != 0) cf_e = d_e[th_idx];

    // handle preconditioners
    if(d_tau != 0)
      tau *= d_tau[th_idx];

    if(invert_step)
      tau = 1. / tau;

    // compute scaled prox argument and step (see e.g. Boyd/Fougner's paper)
    const real arg = cf_a * (d_arg[global_idx] * tau - cf_d) / (cf_e + tau) - cf_b;
    const real step = (cf_e + tau) / (cf_c * cf_a * cf_a);

    // compute prox
    d_result[global_idx] = (prox.Apply(arg, step) + cf_b) / cf_a;
  }
}

Prox1D::Prox1D(
    int index,
    int count,
    const Prox1DCoefficients& coeffs,
    const Prox1DFunction& func)
    : Prox(index, count, 1, false, true), coeffs_(coeffs), func_(func) {

  // might be better to store a, b, c, d, e as an array in the first place
  std::vector<real>* coeff_array[5] = {
    &coeffs_.a,
    &coeffs_.b,
    &coeffs_.c,
    &coeffs_.d,
    &coeffs_.e };
  
  for(int i = 0; i < Prox1DCoefficients::num_coeffs(); i++) {
    const std::vector<real>& cur_elem = *(coeff_array[i]);
    
    assert(!cur_elem.empty());

    real *gpu_data = NULL;
    if(cur_elem.size() > 1) { // if there is more than 1 coeff store it in a global mem.
      assert(cur_elem.size() == count); // sanity check

      cudaMalloc((void **)&gpu_data, sizeof(real) * count);

      // TODO: this assumes linear storage inside a std::vector. is this guaranteed?
      cudaMemcpy(gpu_data, &cur_elem[0], sizeof(real) * count, cudaMemcpyHostToDevice); 
    }

    d_coeffs_.push_back(gpu_data); // for single coeffs, NULL is pushed back.
  }
}

Prox1D::~Prox1D() {
  for(int i = 0; i < Prox1DCoefficients::num_coeffs(); i++) {
    if(d_coeffs_[i])
      cudaFree(d_coeffs_[i]);
  }
}
        
void Prox1D::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step) {
  
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count_ + block.x - 1) / block.x, 1, 1);

  // evil macro but should make things more readable
  #define CALL_PROX1D_KERNEL(Func) \
    Prox1DKernel<Func<real>, real> \
        <<<grid, block>>>( \
             d_arg, \
             d_result, \
             tau, \
             d_tau, \
             Func<real>(), \
             d_coeffs_[0], d_coeffs_[1], d_coeffs_[2], d_coeffs_[3], d_coeffs_[4], \
             coeffs_.a[0], coeffs_.b[0], coeffs_.c[0], coeffs_.d[0], coeffs_.e[0], \
             index_, count_, invert_step)
  
  switch(func_) {    
    case kZero: 
      CALL_PROX1D_KERNEL(Prox1DZero);
      break;
      
    case kAbs: 
      CALL_PROX1D_KERNEL(Prox1DAbs);
      break;

    case kSquare: 
      CALL_PROX1D_KERNEL(Prox1DSquare);
      break;

    case kIndicatorLeq: 
      CALL_PROX1D_KERNEL(Prox1DIndicatorLeq);
      break;

    case kIndicatorEq: 
      CALL_PROX1D_KERNEL(Prox1DIndicatorEq);
      break;

    case kIndicatorAbsLeq: 
      CALL_PROX1D_KERNEL(Prox1DIndicatorAbsLeq);
      break;

    default:
      break;
  }

  cudaDeviceSynchronize();
}

int Prox1D::gpu_mem_amount() {
  int total_mem = 0;
  for(int i = 0; i < Prox1DCoefficients::num_coeffs(); i++) {
    if(d_coeffs_[i] != 0)
      total_mem += count_ * sizeof(real);
  }

  return total_mem;
}
