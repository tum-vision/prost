string(TIMESTAMP VERSION "%Y-%m-%d")
add_definitions(-DPROST_VERSION="0.0.1-build-${VERSION}")

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC -O3)
endif()

if(UNIX)
  list(APPEND CUDA_NVCC_FLAGS -std=c++11)
endif()

set(SOURCES

  "linop/block.cu"
  "linop/block_zero.cu"
  "linop/block_sparse.cu"
  "linop/linearoperator.cu"
  
  "prox/prox.cu"
  "prox/prox_elem_operation.cu"
  "prox/proj_epi_quadratic_fun.cu"
  "prox/prox_moreau.cu"
  "prox/prox_zero.cu"
  
  "backend/backend_pdhg.cu"

  "common.cu"
  "problem.cu"
  "solver.cu"

  "../include/prost/linop/block.hpp"
  "../include/prost/linop/block_zero.hpp"
  "../include/prost/linop/block_sparse.hpp"
  "../include/prost/linop/linearoperator.hpp"

  "../include/prost/prox/prox.hpp"
  "../include/prost/prox/prox_separable_sum.hpp"
  "../include/prost/prox/prox_elem_operation.hpp"
  "../include/prost/prox/proj_epi_quadratic_fun.hpp"
  "../include/prost/prox/prox_moreau.hpp"
  "../include/prost/prox/prox_zero.hpp"
  "../include/prost/prox/shared_mem.hpp"
  "../include/prost/prox/vector.hpp"
  "../include/prost/prox/elemop/elem_operation.hpp"
  "../include/prost/prox/elemop/elem_operation_1d.hpp"
  "../include/prost/prox/elemop/elem_operation_norm2.hpp"
  "../include/prost/prox/elemop/elem_operation_ind_simplex.hpp"
  "../include/prost/prox/elemop/function_1d.hpp"

  "../include/prost/backend/backend.hpp"
  "../include/prost/backend/backend_pdhg.hpp"

  "../include/prost/common.hpp"
  "../include/prost/config.hpp"
  "../include/prost/exception.hpp"
  "../include/prost/problem.hpp"
  "../include/prost/solver.hpp"
)

cuda_add_library(prost STATIC ${SOURCES})
