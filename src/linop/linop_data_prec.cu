#include "linop/linop_data_prec.hpp"
#include <iostream>

template<typename T>
LinOpDataPrec<T>::LinOpDataPrec(size_t row,
                                size_t col,
                                size_t nx,
                                size_t ny,
                                size_t L,
                                T left, T right)
    : LinOp<T>(row, col, nx*ny*(L + 1 + 2*(L-1)), nx*ny*(L + 2*(L-1))), nx_(nx), ny_(ny), L_(L), left_(left), right_(left)
{
}

template<typename T>
__global__
void LinOpDataPrecKernel(T *d_res,
                           T *d_rhs,
                           size_t nx,
                           size_t ny,
                           size_t L,
                           T left,
                           T right)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y_tilde = threadIdx.y + blockDim.y * blockIdx.y;
  size_t y = y_tilde  % ny;
  size_t l = y_tilde / ny;

  if(x >= nx || y >= ny || l >= L)
    return;

  size_t idx_u = y + x * ny + l * nx * ny;
  size_t idx_s = nx*ny*L + L*ny*x + y*L + l;
  size_t idx_w = nx*ny*L + (L-1)*ny*nx + L*ny*x + y*L + l;
  


  T delta_t = (right - left) / (L-1);
  T t = left + l * delta_t;
 

  d_res[idx_u] += d_rhs[idx_u];
  if(l==0) {
    d_res[idx_u] += (1 / delta_t) * (d_rhs[idx_s] + (t+delta_t) * d_rhs[idx_w]);    
  } else if(l==L) {
    d_res[idx_u] += 
        (1 / delta_t) * (-d_rhs[idx_s-1] - 
        (t-delta_t) * d_rhs[idx_w-1]);   
  } else {
    d_res[idx_u] += 
        (1 / delta_t) * (-d_rhs[idx_s-1] + d_rhs[idx_s] - 
        (t-delta_t) * d_rhs[idx_w-1] + (t+delta_t) * d_rhs[idx_w]);
  }
}


template<typename T>
__global__
void LinOpDataPrecAdjointUKernel(T *d_res,
                             T *d_rhs,
                             size_t nx,
                             size_t ny,
                             size_t L, T left, T right)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y_tilde = threadIdx.y + blockDim.y * blockIdx.y;
  size_t y = y_tilde  % ny;
  size_t l = y_tilde / ny;

  if(x >= nx || y >= ny || l >= L)
    return;

  size_t idx = y + x * ny + l * nx * ny;

  d_res[idx] += d_rhs[idx];
}

template<typename T>
__global__
void LinOpDataPrecAdjointSKernel(T *d_res,
                             T *d_rhs,
                             size_t nx,
                             size_t ny,
                             size_t L, T left, T right)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y_tilde = threadIdx.y + blockDim.y * blockIdx.y;
  size_t y = y_tilde  % ny;
  size_t l = y_tilde / ny;

  if(x >= nx || y >= ny || l >= L-1)
    return;

  size_t idx_v = y + x * ny + l * nx * ny;
  size_t idx_v1 = y + x * ny + (l+1) * nx * ny;
  size_t idx_s = nx*ny*L + x*ny*L + y*L + l;
  size_t idx_r = idx_s + nx*ny;
  
  T delta_t = (right - left) / (L-1);
  
  d_res[idx_s] += d_rhs[idx_r] + (1 / delta_t) * (d_rhs[idx_v] - d_rhs[idx_v1]);
}

template<typename T>
__global__
void LinOpDataPrecAdjointWKernel(T *d_res,
                             T *d_rhs,
                             size_t nx,
                             size_t ny,
                             size_t L, T left, T right)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y_tilde = threadIdx.y + blockDim.y * blockIdx.y;
  size_t y = y_tilde  % ny;
  size_t l = y_tilde / ny;

  if(x >= nx || y >= ny || l >= L-1)
    return;

  size_t idx_v = y + x * ny + l * nx * ny;
  size_t idx_v1 = y + x * ny + (l+1) * nx * ny;
  size_t idx_w = nx*ny*L + nx*ny*(L-1) + x*ny*L + y*L + l;
  size_t idx_z = idx_w + nx*ny;
  size_t idx_q = nx*ny*L + x*ny + y;
  
  T delta_t = (right - left) / (L-1);
  T t = left + l * delta_t;
  
  d_res[idx_w] += -d_rhs[idx_q] + d_rhs[idx_z] + (1 / delta_t) * ((t+delta_t) * d_rhs[idx_v] - t*d_rhs[idx_v1]);
}

template<typename T>
LinOpDataPrec<T>::~LinOpDataPrec() {
}

template<typename T>
void LinOpDataPrec<T>::EvalLocalAdd(T *d_res, T *d_rhs) {

  dim3 block(1, 128, 1);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_*L_ + block.y - 1) / block.y,
            1);

  LinOpDataPrecKernel<<<grid, block>>>(d_res, d_rhs, nx_, ny_, L_, left_, right_);
}

template<typename T>
void LinOpDataPrec<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {

  dim3 block(2, 128, 1);
  dim3 gridU((nx_ + block.x - 1) / block.x,
            (ny_*(L_-1) + block.y - 1) / block.y,
            1);
  dim3 gridSW((nx_ + block.x - 1) / block.x,
            (ny_*(L_-1) + block.y - 1) / block.y,
            1);

  LinOpDataPrecAdjointUKernel<T><<<gridU, block>>>(d_res, d_rhs, nx_, ny_, L_, left_, right_);
  LinOpDataPrecAdjointSKernel<T><<<gridSW, block>>>(d_res, d_rhs, nx_, ny_, L_-1, left_, right_);
  LinOpDataPrecAdjointWKernel<T><<<gridSW, block>>>(d_res, d_rhs, nx_, ny_, L_-1, left_, right_);
}

template<typename T>
T LinOpDataPrec<T>::row_sum(size_t row, T alpha) const {
    if(row >= nx_*ny_*L_)
        return 1;

    size_t l = row % L_;

    T delta_t = (right_ - left_) / (L_-1);
    T t = left_ + l * delta_t;


    if(l == 0)
        return 1 + (1 + t + delta_t) / delta_t;
    if(l==L_)
        return 1 + (1 + t - delta_t) / delta_t;

    return 1 + (2*(1+t)) / delta_t;
}
  
template<typename T>
T LinOpDataPrec<T>::col_sum(size_t col, T alpha) const {
    if(col < nx_*ny_*L_)
        return 1;

    T delta_t = (right_ - left_) / (L_-1);
    if(col < nx_*ny_*L_ + nx_*ny_*(L_-1))
        return 1 + (2 / delta_t);

    size_t l = col % (L_-1);
    T t = left_ + l * delta_t;

    return 2 + (t + t + delta_t) / delta_t;
}

template class LinOpDataPrec<float>;
template class LinOpDataPrec<double>;
