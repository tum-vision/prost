#include "prox/prox_epi_max_lin.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include "config.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;

template<typename T>
__global__
void ProxEpiMaxLinKernel(T *d_arg,
                         T *d_res,
                         EpiMaxLinCoeffsDevice<T> coeffs,
                         size_t max_count,
                         size_t count,
                         size_t dim,
                         bool interleaved)
{
  extern __shared__ char sh_mem[];
  // pattern x, t, proj1, proj2, a, u, w, v, pinv
  T* sh_arg = reinterpret_cast<T *>(sh_mem);
  T* x = (T*) (((size_t*)(sh_arg + threadIdx.x * (5*dim + 4*dim*dim + max_count*dim + max_count))) + threadIdx.x * dim);
  T* q = x + dim;
  T* proj1 = q + dim;
  T* proj2 = proj1 + dim;
  T* a = proj2 + dim;
  T* u = a + dim*dim;
  T* w = u + dim*dim;
  T* v = w + dim;
  T* pinv = v + dim*dim;
  T* t = pinv + dim*dim;
  T* b = t + max_count*dim;
  size_t* pattern = (size_t*) (b + max_count);

  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    size_t count_local = coeffs.d_ptr_count[tx];
    size_t index = coeffs.d_ptr_index[tx];

    for(size_t i=0; i < dim; i++) {
      if(interleaved) {
        x[i] = d_arg[tx + count * i];
      } else {
        x[i] = d_arg[dim * tx + i];
      }
    }

    for(size_t i = 0; i < count_local; i++) {
        b[i] = coeffs.d_ptr_b[index + i];
        for(size_t j = 0; j < dim-1; j++) {
            t[j*count_local + i] = coeffs.d_ptr_t[index*(dim-1) + i * (dim-1) + j];
        }
        t[(dim-1)*count_local + i] = -1;
    }

    bool feasible = isFeasible(t, b, x, dim, count_local);
    if(!feasible) {
        T d1 = -1;

        for(size_t n = 1; n <= dim; n++) {
            for(size_t i = 0; i < n; i++) {
                pattern[i] = i;
            }

            bool finished = false;
            while(!finished) {
                for(size_t i = 0; i < n; i++) {
                    q[i] = b[pattern[i]];
                    for(size_t j = 0; j < dim; j++) {
                        a[j*dim + i] = t[j * count_local + pattern[i]];
                    }
                }

                int res = projSubspace(a, u, w, v, pinv, dim, n, dim, q, x, proj2);

                if(isFeasible(t, b, proj2, dim, count_local)) {
                  T d2 = calcEuclDist(x, proj2, dim);
                  if(d1 == -1 || d2 < d1) {
                    T* tmp = proj1;
                    proj1 = proj2;
                    proj2 = tmp;
                    d1 = d2;
                  }
                }           

                finished = true;
                for(int i = n-1; i >= 0; i--) {
                    if(pattern[i] < count_local-n+i) {
                      pattern[i]++;
                      for(int j = i+1; j < n; j++) {
                        pattern[j]=pattern[j-1]+1;
                      }
                      finished = false;
                      break;
                    }
                }
            }
        }

        for(size_t i=0; i < dim; i++) {
          if(interleaved) {
            d_res[tx + count * i] = proj1[i];
          } else {
            d_res[dim * tx + i] = proj1[i];
          }
        }
    } else {
        for(size_t i=0; i<dim; i++) {
          if(interleaved) {
            d_res[tx + count * i] = x[i];
          } else {
            d_res[dim * tx + i] = x[i];
          }
        }
    }
  }
}

template<typename T>
ProxEpiMaxLin<T>::ProxEpiMaxLin(size_t index,
                                size_t count,
                                size_t dim,
                                bool interleaved,
                                const EpiMaxLinCoeffs<T>& coeffs) 
    : Prox<T>(index, count, dim, interleaved, false), coeffs_(coeffs) {
  max_count_ = 0;
  for(int i = 0; i < count; i++) {
    if(coeffs.count[i] > max_count_) {
      max_count_ = coeffs.count[i];
    }
  }
}

template<typename T>
ProxEpiMaxLin<T>::~ProxEpiMaxLin() {
  Release();
}

template<typename T>
bool ProxEpiMaxLin<T>::Init() {
  if((coeffs_.index.size() != this->count_) ||
    (coeffs_.count.size() != this->count_)) {
    std::cout << "count_ doesn't match size of indicies/counts array" << std::endl;
    return false;
  }

  T *d_ptr_T = NULL;

  // copy b and t
  size_t count_lin_funs = coeffs_.index[this->count_-1] + coeffs_.count[this->count_-1];
  if(count_lin_funs != coeffs_.b.size() || count_lin_funs*(this->dim_-1) != coeffs_.t.size()) {
    std::cout << "t and b have wrong size." << std::endl;
    return false;   
  }

  size_t size = count_lin_funs * sizeof(T);

  // copy b
  cudaMalloc((void **)&d_ptr_T, size);
  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }

  cudaMemcpy(d_ptr_T, &coeffs_.b[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_b = d_ptr_T;

  // copy t
  size = size*(this->dim_-1);
  cudaMalloc((void **)&d_ptr_T, size);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }

  cudaMemcpy(d_ptr_T, &coeffs_.t[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  coeffs_dev_.d_ptr_t = d_ptr_T;

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



std::cout << std::endl;

    int count_local = 3;
    int bla[count_local];

    for(int n = 1; n <= 3; n++) {
        for(int i = 0; i < n; i++) {
            bla[i] = i;
        }
        
        bool finished = false;
        while(!finished) {
            
            for(int i = 0; i < n; i++) {
              std::cout << bla[i] << " ";
            }
            std::cout << std::endl;

            finished = true;
            for(int i = n-1; i >= 0; i--) {
                if(bla[i] < count_local-n+i) {
                  bla[i]++;
                  for(int j = i+1; j < n; j++) {
                    bla[j]=bla[j-1]+1;
                  }
                  finished = false;
                  break;
                }
            }
        }
    }
    
  return true;
}

template<typename T>
void ProxEpiMaxLin<T>::Release() {
  cudaFree(coeffs_dev_.d_ptr_b);
  cudaFree(coeffs_dev_.d_ptr_t);
  cudaFree(coeffs_dev_.d_ptr_index);
  cudaFree(coeffs_dev_.d_ptr_count);
}

template<typename T>
void ProxEpiMaxLin<T>::EvalLocal(T *d_arg,
                                    T *d_res,
                                    T *d_tau,
                                    T tau,
                                    bool invert_tau)
{
  dim3 block(64, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  std::cout << block.x << std::endl;
  size_t shmem_bytes = block.x*((5*this->dim_ + 4*this->dim_*this->dim_ + this->max_count_*this->dim_ + this->max_count_)*sizeof(T) + this->dim_*sizeof(size_t));
  ProxEpiMaxLinKernel<T>
      <<<grid, block, shmem_bytes>>>(
          d_arg,
          d_res,
          coeffs_dev_,
          max_count_,
          this->count_,
          this->dim_,
          this->interleaved_);
}

template<typename T>
size_t ProxEpiMaxLin<T>::gpu_mem_amount() {

// TODO
  size_t num_bytes = 0;

  size_t count_t = coeffs_.index[this->count_-1] + coeffs_.count[this->count_-1]*(this->dim_-1);
  size_t size = count_t * sizeof(T);

  num_bytes += size;
  size = this->count_ * sizeof(T);
  num_bytes += 2 * size;
  size = this->count_ * sizeof(size_t);
  num_bytes += 2 * size;

  return num_bytes;
}

// Explicit template instantiation
template class ProxEpiMaxLin<float>;
template class ProxEpiMaxLin<double>;
