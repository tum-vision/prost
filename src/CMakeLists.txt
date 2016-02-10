find_package(CUDA REQUIRED)

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)
endif()

list(APPEND CUDA_NVCC_FLAGS -std=c++11)

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

  "problem.cu"
  "solver.cu"
)

cuda_add_library(prost STATIC ${SOURCES})
