#include "linop/linop_gradient.hpp"

template<typename T>
__global__
void LinOpGradient2DKernel(T *d_rhs,
                           T *d_res,
                           size_t nx,
                           size_t ny,
                           size_t L)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t l = threadIdx.z + blockDim.z * blockIdx.z;

  T gx, gy;
  size_t idx = l + y * L + x * ny * L;

  if(x < nx - 1)
    gx = d_rhs[idx + ny * L] - d_rhs[idx];
  else
    gx = static_cast<T>(0);

  if(y < ny - 1)
    gy = d_rhs[idx + L] - d_rhs[idx];
  else
    gy = static_cast<T>(0);

  d_res[y + x * ny + l * (ny * nx)] += gx;
  d_res[y + x * ny + l * (ny * nx) + nx * ny * L] += gy;
}


template<typename T>
__global__
void LinOpDivergence2DKernel(T *d_rhs,
                             T *d_res,
                             size_t nx,
                             size_t ny,
                             size_t L)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t l = threadIdx.z + blockDim.z * blockIdx.z;

  T divx, divy;
  size_t idx = l + y * L + x * ny * L;

  divx = d_rhs[idx];
  divy = d_rhs[idx + nx * ny * L];

  if(x > 0)
    divx -= d_rhs[idx - ny * L];

  if(y > 0)
    divy -= d_rhs[idx - L];

  d_res[idx] -= (divx + divy); // adjoint is minus the divergence
}

template<typename T>
__global__
void LinOpGradient3DKernel(T *d_rhs,
                           T *d_res,
                           size_t nx,
                           size_t ny,
                           size_t L)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t l = threadIdx.z + blockDim.z * blockIdx.z;

  T gx, gy, gl;
  size_t idx = l + y * L + x * ny * L;

  if(x < nx - 1)
    gx = d_rhs[idx + ny * L] - d_rhs[idx];
  else
    gx = static_cast<T>(0);

  if(y < ny - 1)
    gy = d_rhs[idx + L] - d_rhs[idx];
  else
    gy = static_cast<T>(0);

  if(L < L - 1)
    gl = d_rhs[idx + 1] - d_rhs[idx];
  else
    gl = static_cast<T>(0);

  d_res[y + x * ny + l * (ny * nx)] += gx;
  d_res[y + x * ny + l * (ny * nx) + nx * ny * L] += gy;
  d_res[y + x * ny + l * (ny * nx) + 2 * nx * ny * L] += gl;
}


template<typename T>
__global__
void LinOpDivergence3DKernel(T *d_rhs,
                             T *d_res,
                             size_t nx,
                             size_t ny,
                             size_t L)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t l = threadIdx.z + blockDim.z * blockIdx.z;

  T divx, divy, divl;
  size_t idx = l + y * L + x * ny * L;

  divx = d_rhs[idx];
  divy = d_rhs[idx + nx * ny * L];
  divl = d_rhs[idx + 2 * nx * ny * L];

  if(x > 0)
    divx -= d_rhs[idx - ny * L];

  if(y > 0)
    divy -= d_rhs[idx + nx * ny * L - L];

  if(l > 0)
    divl -= d_rhs[idx + 2 * nx * ny * L - 1];

  d_res[idx] -= (divx + divy + divl); // adjoint is minus the divergence
}

template<typename T>
LinOpGradient2D<T>::LinOpGradient2D(size_t row, size_t col, size_t nx, size_t ny, size_t L)
    : LinOp<T>(row, col, nx*ny*L*2, nx*ny*L), nx_(nx), ny_(ny), L_(L)
{
}

template<typename T>
LinOpGradient2D<T>::~LinOpGradient2D() {
}

template<typename T>
void LinOpGradient2D<T>::EvalLocalAdd(T *d_res, T *d_rhs) {

  dim3 block(16, 16, 8);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_ + block.y - 1) / block.y,
            (L_ + block.z - 1) / block.z);

  LinOpGradient2DKernel<<<grid, block>>>(d_rhs, d_res, nx_, ny_, L_);
}

template<typename T>
void LinOpGradient2D<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  
  dim3 block(16, 16, 8);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_ + block.y - 1) / block.y,
            (L_ + block.z - 1) / block.z);

  LinOpDivergence2DKernel<<<grid, block>>>(d_rhs, d_res, nx_, ny_, L_);
}

template<typename T>
LinOpGradient3D<T>::LinOpGradient3D(size_t row, size_t col, size_t nx, size_t ny, size_t L)
    : LinOp<T>(row, col, nx*ny*L*3, nx*ny*L), nx_(nx), ny_(ny), L_(L)
{
}

template<typename T>
LinOpGradient3D<T>::~LinOpGradient3D() {
}

template<typename T>
void LinOpGradient3D<T>::EvalLocalAdd(T *d_res, T *d_rhs) {

  dim3 block(16, 16, 8);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_ + block.y - 1) / block.y,
            (L_ + block.z - 1) / block.z);

  LinOpGradient3DKernel<<<grid, block>>>(d_rhs, d_res, nx_, ny_, L_);
}

template<typename T>
void LinOpGradient3D<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  
  dim3 block(16, 16, 8);
  dim3 grid((nx_ + block.x - 1) / block.x,
            (ny_ + block.y - 1) / block.y,
            (L_ + block.z - 1) / block.z);

  LinOpDivergence3DKernel<<<grid, block>>>(d_rhs, d_res, nx_, ny_, L_);
}

template class LinOpGradient2D<float>;
template class LinOpGradient2D<double>;
template class LinOpGradient3D<float>;
template class LinOpGradient3D<double>;
