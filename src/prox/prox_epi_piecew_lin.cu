#include "prox/prox_epi_piecew_lin.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include "config.hpp"
#include <iostream>

template<typename T>
__global__
void ProxEpiPiecewLinKernel(T *d_arg,
                            T *d_res,
                            EpiPiecewLinCoeffsDevice<T> coeffs,
                            size_t count,
                            bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  /**if(tx < count) {
    T result[2];

    // get v = (x0, y0) and alpha,beta and count,index
    T alpha = coeffs.d_ptr_alpha[tx];
    T beta = coeffs.d_ptr_beta[tx];
    size_t count = coeffs.d_ptr_count[tx];
    size_t index = coeffs.d_ptr_index[tx];

    T v[2];
    if(interleaved) {
      v[0] = d_arg[tx * 2 + 0];
      v[1] = d_arg[tx * 2 + 1];
    } else {
      v[0] = d_arg[tx + count * 0];
      v[1] = d_arg[tx + count * 1];
    }
    
    // compute vector normal to slope for feasibility-check
    T n_slope[2];
    n_slope[0] = alpha;
    n_slope[1] = -1;

    T x1 = coeffs.d_ptr_x[index];
    T y1 = coeffs.d_ptr_y[index];
    T p[2];
    p[0] = x1;
    p[1] = y1;
    bool feasible = PointInHalfspace(v, p, n_slope, 2);

    T n_halfspace[2];
    n_halfspace[0] = 1;
    n_halfspace[1] = alpha;
    bool halfspace_1 = PointInHalfspace(v, p, n_halfspace, 2);

    bool projected = false;

    if(!feasible && halfspace_1) {
        // point is not feasible wrt to 0-th piece and
        //  lies in rectangle => projection is the 
        //  respective half space projection
        T t = x1*n_slope[0] + y1*n_slope[1];
        ProjectHalfspace(v, n_slope, t, result, 2);
        projected = true;
    }

    if(!projected) {
      for(size_t i = 0; i < count-1; i++) {
        // read "knick" at i+1
        T x2 = coeffs.d_ptr_x[index+i+1];
        T y2 = coeffs.d_ptr_y[index+i+1];

        // compute slope
        T c = (x2-x1) / (y2-y1);

        // compute vector normal to slope
        n_slope[0] = c;
        n_slope[1] = -1;
      
        // check whether point v is feasible wrt i-th piece
        feasible = PointInHalfspace(v, p, n_slope, 2);

        n_halfspace[0] = -1;
        n_halfspace[1] = -c;

        bool halfspace_2 = PointInHalfspace(v, p, n_halfspace, 2);

        if(!feasible) {
          // point is not feasible wrt to i-th piece
          if(!halfspace_1 && !halfspace_2) {
            // point lies in (i-1)-th normal cone => projection is the "knick"
            result[0] = x1;
            result[1] = y1; 
            projected = true;
            break;
          }

          // compute inverse normal -n s.t. the two normals n and -n
          //  together with the two knicks define a reactangle
          n_halfspace[0] = -n_halfspace[0];
          n_halfspace[1] = -n_halfspace[1];
          
          // check wether point lies in i-th halfspace
          p[0] = x2;
          p[1] = y2;
          halfspace_1 = PointInHalfspace(v, p, n_halfspace, 2);
          if(halfspace_2 && halfspace_1) {
            // point lies in i-th rectangle => projection is the 
            //  respective half space projection
            T t = x1*n_slope[0] + y1*n_slope[1];
            ProjectHalfspace(v, n_slope, t, result, 2);
            projected = true;
            break;
          }
        }
 
        // hand over variables for next iteration
        x1 = x2;
        y1 = y2;
      }
    }

    if(!projected) {
      // compute vector normal to slope
      n_slope[0] = beta;
      n_slope[1] = -1;
      
      // check whether point v is feasible wrt i-th piece
      feasible = PointInHalfspace(v, p, n_slope, 2);

      n_halfspace[0] = -1;
      n_halfspace[1] = -beta;

      bool halfspace_2 = PointInHalfspace(v, p, n_halfspace, 2);

      if(!feasible) {
        // point is not feasible wrt to i-th piece
        if(!halfspace_1 && !halfspace_2) {
          // point lies in last normal cone => projection is the last "knick"
          result[0] = x1;
          result[1] = y1; 
          projected = true;
        } else if(halfspace_2) {
          // point lies in last rectangle => projection is the 
          //  respective half space projection
          T t = x1*n_slope[0] + y1*n_slope[1];
          ProjectHalfspace(v, n_slope, t, result, 2);
          projected = true;
        }
      }
    }
   
    // point has not been projected. That means we output the original point    
    if(!projected) {
      result[0] = v[0];
      result[1] = v[1];      
    }
    
    // write out result
    if(interleaved) {
      d_res[tx * 2 + 0] = result[0];
      d_res[tx * 2 + 1] = result[1];
    }
    else {
      d_res[tx + count * 0] = result[0];
      d_res[tx + count * 1] = result[1];
    }
  }**/
}

template<typename T>
ProxEpiPiecewLin<T>::ProxEpiPiecewLin(size_t index,
                                      size_t count,
                                      bool interleaved,
                                      const EpiPiecewLinCoeffs<T>& coeffs)
    
    : Prox<T>(index, count, 2, interleaved, false), coeffs_(coeffs)
{
}

template<typename T>
ProxEpiPiecewLin<T>::~ProxEpiPiecewLin() {
  Release();
}

template<typename T>
bool ProxEpiPiecewLin<T>::Init() {
  
  std::cout << "Init" <<std::endl;
  if(coeffs_.x.empty() || coeffs_.y.empty() 
    || coeffs_.alpha.empty() || coeffs_.beta.empty() || 
       coeffs_.index.empty() || coeffs_.count.empty())
    return false;

  for(int i = 0; i < this->count_; i++) {
    std::cout << coeffs_.index[i]<< "  => " <<coeffs_.count[i]<<std::endl;
  }
  /**
  // copy x and y
  size_t count_xy = coeffs_.index[this->count_-1] + coeffs_.count[this->count_-1];

  // copy x
  T *d_ptr_x = NULL;
  cudaMalloc((void **)&d_ptr_x, count_xy * sizeof(T));
  if(cudaGetLastError() != cudaSuccess)
    return false;

  cudaMemcpy(d_ptr_x, &coeffs_.x[0], sizeof(T) * count_xy, cudaMemcpyHostToDevice);
  coeffs_dev_.d_ptr_x = d_ptr_x;

  // copy y
  T *d_ptr_y = NULL;
  cudaMalloc((void **)&d_ptr_y, count_xy * sizeof(T));
  if(cudaGetLastError() != cudaSuccess)
    return false;

  cudaMemcpy(d_ptr_y, &coeffs_.y[0], sizeof(T) * count_xy, cudaMemcpyHostToDevice);
  coeffs_dev_.d_ptr_y = d_ptr_y;

  // copy alpha
  T *d_ptr_alpha = NULL;
  cudaMalloc((void **)&d_ptr_alpha, this->count_ * sizeof(T));
  if(cudaGetLastError() != cudaSuccess)
    return false;

  cudaMemcpy(d_ptr_alpha, &coeffs_.alpha[0], sizeof(T) * this->count_, cudaMemcpyHostToDevice);
  coeffs_dev_.d_ptr_alpha = d_ptr_alpha;


  // copy beta
  T *d_ptr_beta = NULL;
  cudaMalloc((void **)&d_ptr_beta, this->count_ * sizeof(T));
  if(cudaGetLastError() != cudaSuccess)
    return false;

  cudaMemcpy(d_ptr_beta, &coeffs_.beta[0], sizeof(T) * this->count_, cudaMemcpyHostToDevice);
  coeffs_dev_.d_ptr_beta = d_ptr_beta;


  // copy count
  size_t *d_ptr_count = NULL;
  cudaMalloc((void **)&d_ptr_count, this->count_ * sizeof(size_t));
  if(cudaGetLastError() != cudaSuccess)
    return false;

  cudaMemcpy(d_ptr_count, &coeffs_.count[0], sizeof(size_t) * this->count_, cudaMemcpyHostToDevice);
  coeffs_dev_.d_ptr_count = d_ptr_count;

  // copy index
  size_t *d_ptr_index = NULL;
  cudaMalloc((void **)&d_ptr_index, this->index_ * sizeof(size_t));
  if(cudaGetLastError() != cudaSuccess)
    return false;

  cudaMemcpy(d_ptr_index, &coeffs_.index[0], sizeof(size_t) * this->count_, cudaMemcpyHostToDevice);
  coeffs_dev_.d_ptr_index = d_ptr_index;
**/
  std::cout << "Copy success" <<std::endl;

  return true;
}

template<typename T>
void ProxEpiPiecewLin<T>::Release() {
  cudaFree(coeffs_dev_.d_ptr_x);
  cudaFree(coeffs_dev_.d_ptr_y);
  cudaFree(coeffs_dev_.d_ptr_alpha);
  cudaFree(coeffs_dev_.d_ptr_beta);
  cudaFree(coeffs_dev_.d_ptr_index);
  cudaFree(coeffs_dev_.d_ptr_count);

}

template<typename T>
void ProxEpiPiecewLin<T>::EvalLocal(T *d_arg,
                                    T *d_res,
                                    T *d_tau,
                                    T tau,
                                    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

   std::cout << "Eval" <<std::endl;
  /**ProxEpiPiecewLinKernel<T>
      <<<grid, block>>>(
          d_arg,
          d_res,
          coeffs_dev_,
          this->count_,
          this->interleaved_);**/
}

// Explicit template instantiation
template class ProxEpiPiecewLin<float>;
template class ProxEpiPiecewLin<double>;
