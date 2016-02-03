#ifndef MEX_CONFIG_HPP_
#define MEX_CONFIG_HPP_

// fixes mex compilation bugs for CUDA code
#define __device__
#define __host__
#define __syncthreads
#define __shared__

typedef float real;
//typedef double real;

#endif
