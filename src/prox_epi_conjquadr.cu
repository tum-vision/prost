#include "prox_epi_conjquadr.hpp"

#include <cassert>

template<typename real>
__global__
void ProxEpiConjQuadrKernel(
    const real *d_arg, // (x0, y0)
    real *d_result, // projection of (x0, y0) onto the Epigraph
    const real *d_a, 
    const real *d_b,
    const real *d_c,
    const real *d_alpha,
    const real *d_beta,
    int index,
    int count,
    bool interleaved)
{
  int th_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(th_idx < count) {
    real result[2];

    // get v = (x0, y0) and a,b,c,alpha,beta
    real a, b, c, alpha, beta;
    real v[2];

    a = d_a[th_idx];
    b = d_b[th_idx];
    c = d_c[th_idx];
    alpha = d_alpha[th_idx];
    beta = d_beta[th_idx];

    if(interleaved) {
      v[0] = d_arg[index + th_idx * 2 + 0];
      v[1] = d_arg[index + th_idx * 2 + 1];
    }
    else {
      v[0] = d_arg[index + th_idx + count * 0];
      v[1] = d_arg[index + th_idx + count * 1];
    }
    
    // check which case applies (0 = A, 1 = B, 2 = C)
    const real p_A[2] = { 2 * a * alpha + b, a * alpha * alpha - c };
    const real p_B[2] = { 2 * a * beta + b, a * beta * beta - c };
    real n_A[2] = { 1, alpha };
    real n_B[2] = { -1, -beta }; 
    
    int proj_case;
    if(PointInHalfspace(v, p_A, n_A, 2))
      proj_case = 0;
    else if(PointInHalfspace(v, p_B, n_B, 2))
      proj_case = 2;
    else
      proj_case = 1;
    
    // perform projection
    switch(proj_case) {
      case 0: { // case A
        n_A[0] = -alpha;
        n_A[1] = 1.;
        const real t = -a * alpha * alpha - b * alpha - c;
          
        ProjectHalfspace<real>(
            v,
            n_A,
            t,
            result,
            2);
      } break;

      case 1: { // case B
        if(a > 0) 
          ProjectParabolaGeneral<real>(
              v[0],
              v[1],
              1. / (4. * a),
              -b / (2. * a),
              b * b / (4. * a) - c,
              result[0],
              result[1]);
        else {
          // if a <= 0 the parabola disappears and we're in the normal cone.
          result[0] = a * (alpha + beta) + b;
          result[1] = alpha * beta * a - c;
        }
          
      } break;

      case 2: { // case C
        n_B[0] = -beta;
        n_B[1] = 1.;
        const real t = -a * beta * beta - b * beta - c;

        ProjectHalfspace<real>(
            v,
            n_B,
            t,
            result,
            2);
      } break;
    }      
    
    // write result
    if(interleaved) {
      d_result[index + th_idx * 2 + 0] = result[0];
      d_result[index + th_idx * 2 + 1] = result[1];
    }
    else {
      d_result[index + th_idx + count * 0] = result[0];
      d_result[index + th_idx + count * 1] = result[1];
    }
  }
}

ProxEpiConjQuadr::ProxEpiConjQuadr(
    int index,
    int count,
    bool interleaved,
    const EpiConjQuadrCoeffs& coeffs)
    
    : Prox(index, count, 2, interleaved, false), coeffs_(coeffs)
{
  assert(EpiConjQuadrCoeffs::num_coeffs() == 5);
  
  std::vector<real>* coeff_array[5] = {
    &coeffs_.a,
    &coeffs_.b,
    &coeffs_.c,
    &coeffs_.alpha,
    &coeffs_.beta };
  
  for(int i = 0; i < 5; i++) {
    const std::vector<real>& cur_elem = *(coeff_array[i]);
    assert(!cur_elem.empty());
    
    real *ptr;
    cudaMalloc((void **)&ptr, count * sizeof(real));
    d_coeffs_.push_back(ptr);
    cudaMemcpy(ptr, &cur_elem[0], sizeof(real) * count, cudaMemcpyHostToDevice);
  }
}

ProxEpiConjQuadr::~ProxEpiConjQuadr() {
  for(int i = 0; i < d_coeffs_.size(); i++) {
    cudaFree(d_coeffs_[i]);
  }
}

void ProxEpiConjQuadr::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count_ + block.x - 1) / block.x, 1, 1);

  ProxEpiConjQuadrKernel<real>
      <<<grid, block>>>(
          d_arg,
          d_result,
          d_coeffs_[0],
          d_coeffs_[1],
          d_coeffs_[2],
          d_coeffs_[3],
          d_coeffs_[4],
          index_,
          count_,
          interleaved_);
}
