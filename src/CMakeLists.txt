string(TIMESTAMP VERSION "%Y-%m-%d")

include(../cmake/CudaVersion.cmake)
prost_select_nvcc_arch_flags(PROST_NVCC_FLAGS)
#message(${PROST_NVCC_FLAGS_readable})
message("Detected NVCC flags:${PROST_NVCC_FLAGS}")

set(PROST_VERSION "0.2-build-${VERSION}")
add_definitions(-DPROST_VERSION=${PROST_VERSION})

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC -O3)
endif()

if(UNIX)
  list(APPEND CUDA_NVCC_FLAGS -std=c++11)
endif()

#list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_20,code=sm_21")
list(APPEND CUDA_NVCC_FLAGS ${PROST_NVCC_FLAGS})

set(SOURCES

  "linop/block.cu"
  "linop/block_dense.cu"
  "linop/block_diags.cu"
  "linop/block_gradient2d.cu"
  "linop/block_gradient3d.cu"
  "linop/block_id_kron_sparse.cu"
  "linop/block_sparse.cu"
  "linop/block_sparse_kron_id.cu"
  "linop/block_zero.cu"
  "linop/dual_linearoperator.cu"
  "linop/linearoperator.cu"
  
  "prox/prox.cu"
  "prox/prox_elem_operation.cu"
  "prox/prox_ind_epi_quad.cu"
  "prox/prox_ind_epi_polyhedral.cu"
  "prox/prox_moreau.cu"
  "prox/prox_transform.cu"
  "prox/prox_zero.cu"
  
  "backend/backend_pdhg.cu"
  "backend/backend_admm.cu"

  "common.cu"
  "problem.cu"
  "solver.cu"

  "../include/prost/linop/block.hpp"
  "../include/prost/linop/block_dense.hpp"
  "../include/prost/linop/block_diags.hpp"
  "../include/prost/linop/block_gradient2d.hpp"
  "../include/prost/linop/block_gradient3d.hpp"
  "../include/prost/linop/block_id_kron_sparse.hpp"
  "../include/prost/linop/block_sparse.hpp"
  "../include/prost/linop/block_sparse_kron_id.hpp"
  "../include/prost/linop/block_zero.hpp"
  "../include/prost/linop/dual_linearoperator.hpp"
  "../include/prost/linop/linearoperator.hpp"

  "../include/prost/prox/prox.hpp"
  "../include/prost/prox/prox_separable_sum.hpp"
  "../include/prost/prox/prox_elem_operation.hpp"
  "../include/prost/prox/prox_ind_epi_quad.hpp"
  "../include/prost/prox/prox_ind_epi_polyhedral.hpp"
  "../include/prost/prox/prox_moreau.hpp"
  "../include/prost/prox/prox_transform.hpp"
  "../include/prost/prox/prox_zero.hpp"
  "../include/prost/prox/shared_mem.hpp"
  "../include/prost/prox/vector.hpp"
  "../include/prost/prox/helper.hpp"
  "../include/prost/prox/elemop/elem_operation.hpp"
  "../include/prost/prox/elemop/elem_operation_1d.hpp"
  "../include/prost/prox/elemop/elem_operation_norm2.hpp"
  "../include/prost/prox/elemop/elem_operation_ind_simplex.hpp"
  "../include/prost/prox/elemop/elem_operation_singular_nx2.hpp"
  "../include/prost/prox/elemop/function_1d.hpp"
  "../include/prost/prox/elemop/function_2d.hpp"

  "../include/prost/backend/backend.hpp"
  "../include/prost/backend/backend_pdhg.hpp"
  "../include/prost/backend/backend_admm.hpp"

  "../include/prost/common.hpp"
  "../include/prost/config.hpp"
  "../include/prost/exception.hpp"
  "../include/prost/problem.hpp"
  "../include/prost/solver.hpp"
)

cuda_add_library(prost STATIC ${SOURCES} ${PROST_CUSTOM_SOURCES})
