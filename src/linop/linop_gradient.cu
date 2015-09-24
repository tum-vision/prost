
template<typename T>
__global__
void LinOpGradientKernel(T *d_rhs,
                         T *d_res,
                         size_t nx,
                         size_t ny,
                         size_t L)
{
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t L = threadIdx.z + blockDim.z * blockIdx.z;

  // compute gradient ...
  float gx, gy;
  size_t idx = L + y * L + x * ny * L;

  gx = d_rhs[idx1] - d_rhs[idx];


  d_res[y + x * ny + L * (ny * nx)] = gx;
  d_res[y + x * ny + L * (ny * nx) + nx * ny * L] = gy;
}


template<typename T>
__global__
void LinOpMinusDivergenceKernel(T *d_rhs,
                                T *d_res,
                                size_t nx,
                                size_t ny,
                                size_t L)


void LinOpGradient::EvalLocalAdd(T *d_res, T *d_rhs) {

  dim3 block(16, 16, 4);
  dim3 grid((nx + block.x - 1)/ block.x, (ny + block.y - 1) / block.y, (L + block.z - 1) / block.z);

  LinOpGradientKernel<<<grid, block>>>(d_rhs, d_res, nx_, ny_, L_);
}

void LinOpGradient::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  
  LinOpMinusDivergenceKernel<<<grid, block>>>(d_rhs, d_res, nx_, ny_, L_);
}