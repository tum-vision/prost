#include "prox/prox_epi_piecew_lin.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include "config.hpp"
#include <iostream>

using namespace std;

template<typename T>
__global__
void ProxEpiPiecewLinKernel(T *d_arg,
                            T *d_res,
                            EpiPiecewLinCoeffsDevice<T> coeffs,
                            size_t count,
                            bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    T result[2];

    // get v = (x0, y0) and alpha,beta and count,index
    T alpha = coeffs.d_ptr_alpha[tx];
    T beta = coeffs.d_ptr_beta[tx];
    size_t count_local = coeffs.d_ptr_count[tx];
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

    bool feasible_left = PointInHalfspace(v, p, n_slope, 2);
    
    T n_halfspace[2];
    n_halfspace[0] = 1;
    n_halfspace[1] = alpha;

    bool halfspace_left = PointInHalfspace(v, p, n_halfspace, 2);

    bool projected = false;

    if(!feasible_left && halfspace_left) {
      // point is not feasible wrt to 0-th piece and
      //  lies in rectangle => projection is the 
      //  respective half space projection

      T t = x1*n_slope[0] + y1*n_slope[1];
      ProjectHalfspace(v, n_slope, t, result, 2);
      projected = true;
    }

    if(!projected) {
      for(size_t i = 0; i < count_local-1; i++) {
        // read "knick" at i+1
        T x2 = coeffs.d_ptr_x[index+i+1];
        T y2 = coeffs.d_ptr_y[index+i+1];

        // compute slope
        T c = (y2-y1) / (x2-x1);

        // compute vector normal to slope
        n_slope[0] = c;
        n_slope[1] = -1;

        // check whether point v is feasible wrt i-th piece
        bool feasible_right = PointInHalfspace(v, p, n_slope, 2);

        n_halfspace[0] = -1;
        n_halfspace[1] = -c;


        bool halfspace_right = PointInHalfspace(v, p, n_halfspace, 2);

        p[0] = x2;
        p[1] = y2;
        if(!feasible_left || !feasible_right) {
          // point is not feasible wrt to i-th piece or (i-1)-th piece
          if(!halfspace_left && !halfspace_right) {
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
          halfspace_left = PointInHalfspace(v, p, n_halfspace, 2);
          if(halfspace_right && halfspace_left) {
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
        feasible_left = feasible_right;
      }
    }

 

    if(!projected) {
      // compute vector normal to slope
      n_slope[0] = beta;
      n_slope[1] = -1; 

      // check whether point v is feasible wrt i-th piece
      bool feasible_right = PointInHalfspace(v, p, n_slope, 2);

      n_halfspace[0] = -1;
      n_halfspace[1] = -beta;

      bool halfspace_right = PointInHalfspace(v, p, n_halfspace, 2);

      if(!feasible_left || !feasible_right) {
        // point is not feasible wrt to i-th piece or (i-1)-th piece
        if(!halfspace_left && !halfspace_right) {
          // point lies in last normal cone => projection is the last "knick"
          result[0] = x1;
          result[1] = y1; 

          projected = true;
        } else if(halfspace_right) {
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
    } else {
      d_res[tx + count * 0] = result[0];
      d_res[tx + count * 1] = result[1];
    }
  }
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
  if(coeffs_.x.empty() || coeffs_.y.empty() 
    || coeffs_.alpha.empty() || coeffs_.beta.empty() || 
       coeffs_.index.empty() || coeffs_.count.empty())
    return false;  

  if((coeffs_.index.size() != this->count_) ||
    (coeffs_.count.size() != this->count_))
  {
    cout << "count_ doesn't match size of indicies/counts array" << endl;
    return false;
  }

  // Ensure convexity
  for(int i = 0; i < this->count_; i++) {
    T slope_left = coeffs_.alpha[i];
    for(int j = coeffs_.index[i]; j < coeffs_.index[i] + coeffs_.count[i] - 1; j++) {
      T slope_right = (coeffs_.y[j+1]-coeffs_.y[j]) / (coeffs_.x[j+1]-coeffs_.x[j]);
      if(slope_right < slope_left) {
        std::cout <<std::endl<< "Error: Non-convex energy:" << i <<std::endl;
        cout << "slope_left=" << slope_left << ", slope_right=" << slope_right << endl;
        return false;
      }
      slope_left = slope_right;
    }
    if(coeffs_.beta[i] < slope_left) {
      std::cout << "Error: Non-convex energy (right bdry):" << i << std::endl;
      cout << "slope_left=" << slope_left << ", slope_right=" << coeffs_.beta[i] << endl;
      return false;
    }
  }

  T *d_ptr_T = NULL;

/*
  cout << "index.size() = " << coeffs_.index.size() << endl;
  cout << "count.size() = " << coeffs_.count.size() << endl;

  cout << "coeffs_.index(end) = " << coeffs_.index[this->count_ - 1] << endl;
  cout << "coeffs_.count(end) = " << coeffs_.count[this->count_ - 1] << endl;
*/

  // copy x and y
  size_t count_xy = coeffs_.index[this->count_-1] + coeffs_.count[this->count_-1];
  size_t size = count_xy * sizeof(T);
/*
  cout << "this->count_=" << this->count_ << endl;
  cout << "count_xy=" << count_xy << endl;

  return false;
*/
  // copy x
  cudaMalloc((void **)&d_ptr_T, size);
  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  cudaMemcpy(d_ptr_T, &coeffs_.x[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_x = d_ptr_T;

  // copy y
  cudaMalloc((void **)&d_ptr_T, size);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }

  cudaMemcpy(d_ptr_T, &coeffs_.y[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_y = d_ptr_T;

  // copy alpha
  size = this->count_ * sizeof(T);
  cudaMalloc((void **)&d_ptr_T, size);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  cudaMemcpy(d_ptr_T, &coeffs_.alpha[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_alpha = d_ptr_T;


  // copy beta
  cudaMalloc((void **)&d_ptr_T, size);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }

  cudaMemcpy(d_ptr_T, &coeffs_.beta[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_beta = d_ptr_T;


  // copy count
  size = this->count_ * sizeof(size_t);

  size_t *d_ptr_size_t = NULL;
  cudaMalloc((void **)&d_ptr_size_t, size);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }

  cudaMemcpy(d_ptr_size_t, &coeffs_.count[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_count = d_ptr_size_t;

  // copy index
  cudaMalloc((void **)&d_ptr_size_t, size);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }

  cudaMemcpy(d_ptr_size_t, &coeffs_.index[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_index = d_ptr_size_t;

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

  ProxEpiPiecewLinKernel<T>
      <<<grid, block>>>(
          d_arg,
          d_res,
          coeffs_dev_,
          this->count_,
          this->interleaved_);
}

// Explicit template instantiation
template class ProxEpiPiecewLin<float>;
template class ProxEpiPiecewLin<double>;
