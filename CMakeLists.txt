cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

# Platform specific setup
if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang++)
  set(CMAKE_CXX_FLAGS "-O3 -stdlib=libstdc++ -std=c++11 -Wall")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -fPIC -Wall")
elseif(WIN32)
  # TODO: compile on windows
else()
  message( FATAL_ERROR "Operating system not supported." )
endif()

# TODO: build fat binary for several common architectures
set(CUDA_NVCC_FLAGS "-O3 -std=c++11 -gencode=arch=compute_52,code=compute_52")

include_directories("include")

set(SOURCES
  "src/linop/block.cu"
  "src/linop/block_zero.cu"
  "src/linop/block_sparse.cu"
  "src/linop/linearoperator.cu"

  "src/prox/prox.cu"
  "src/prox/prox_elem_operation.cu"
  "src/prox/prox_moreau.cu"
  "src/prox/prox_zero.cu"

  "src/backend/backend_pdhg.cu"

  "src/problem.cu"
  "src/solver.cu"

  "include/linop/block.hpp"
  "include/linop/block_zero.hpp"
  "include/linop/block_sparse.hpp"
  "include/linop/linearoperator.hpp"

  "include/prox/prox.hpp"
  "include/prox/prox_moreau.hpp"
  "include/prox/prox_zero.hpp"

  "include/backend/backend.hpp"
  "include/backend/backend_pdhg.hpp"

  "include/exception.hpp"
  "include/factory.hpp"
  "include/solver.hpp"
  "include/problem.hpp"
)
  
cuda_add_library(pdsolver STATIC ${SOURCES})
