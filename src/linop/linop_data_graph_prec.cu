#include "linop/linop_data_graph_prec.hpp"
#include <iostream>

template<typename T>
LinOpDataGraphPrec<T>::LinOpDataGraphPrec(size_t row,
                                size_t col,
                                size_t nx,
                                size_t ny,
                                size_t L,
                                T t_min, T t_max)
    : LinOp<T>(row, col, nx*ny*(L-1), nx*ny*(L-1)), nx_(nx), ny_(ny), L_(L), t_min_(t_min), t_max_(t_max)
{
}

template<typename T>
__global__
void LinOpDataGraphPrecKernel(T *d_res,
                           T *d_rhs,
                           size_t nx,
                           size_t ny,
                           size_t L,
                           T t_min,
                           T t_max)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y_tilde = threadIdx.y + blockDim.y * blockIdx.y;
  size_t l = y_tilde  % (L-1);
  size_t y = y_tilde / (L-1);
  
  if(x >= nx || y >= ny || l >= L-1)
    return;
  
  size_t idx = (L-1)*ny*x + y*(L-1) + l;
  
  T delta_t = (t_max - t_min) / (L-1);
  T t = t_min + l * delta_t;
  
  T add_v = -t*d_rhs[idx];
  for(size_t i = 1; i < L-1-l; i++) {
    add_v += delta_t*d_rhs[idx+i];
  }
  
  d_res[idx] += add_v;
}

template<typename T>
__global__
void LinOpDataGraphPrecAdjointKernel(T *d_res,
                             T *d_rhs,
                             size_t nx,
                             size_t ny,
                             size_t L, T t_min, T t_max)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y_tilde = threadIdx.y + blockDim.y * blockIdx.y;
  size_t l = y_tilde  % (L-1);
  size_t y = y_tilde / (L-1);
  
  if(x >= nx || y >= ny || l >= L-1)
    return;

  size_t idx = (L-1)*ny*x + y*(L-1) + l;
  
  T delta_t = (t_max - t_min) / (L-1);
  T t = t_min + l * delta_t;

  T add_s = -t*d_rhs[idx];
  for(size_t i = 1; i <= l; i++) {
    add_s += delta_t*d_rhs[idx-i];
  }
  
  d_res[idx] += add_s;
}

template<typename T>
LinOpDataGraphPrec<T>::~LinOpDataGraphPrec() {
}

template<typename T>
void LinOpDataGraphPrec<T>::EvalLocalAdd(T *d_res, T *d_rhs) {
  dim3 block(1, 128, 1);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_*(L_-1) + block.y - 1) / block.y,
            1);

  LinOpDataGraphPrecKernel<<<grid, block>>>(d_res, d_rhs, nx_, ny_, L_, t_min_, t_max_);
}

template<typename T>
void LinOpDataGraphPrec<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {

  dim3 block(1, 128, 1);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_*(L_-1) + block.y - 1) / block.y,
            1);

  LinOpDataGraphPrecAdjointKernel<T><<<grid, block>>>(d_res, d_rhs, nx_, ny_, L_, t_min_, t_max_);
}

template<typename T>
T LinOpDataGraphPrec<T>::row_sum(size_t row, T alpha) const {
  size_t l = row % (L_-1);
  T delta_t = (t_max_ - t_min_) / (L_-1);
  T t = t_min_ + l * delta_t;
  return t + (L_-2-l)*delta_t;  
}
  
template<typename T>
T LinOpDataGraphPrec<T>::col_sum(size_t col, T alpha) const {
  size_t l = col % (L_-1);
  T delta_t = (t_max_ - t_min_) / (L_-1);
  T t = t_min_ + l * delta_t;
  return t + l*delta_t;
}

template class LinOpDataGraphPrec<float>;
template class LinOpDataGraphPrec<double>;
