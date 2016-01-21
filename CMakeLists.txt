cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

# Platform specific setup
if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
  set(CMAKE_CXX_FLAGS "-O3 -stdlib=libstdc++ -std=c++11 -Wall")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -fPIC -Wall")
elseif(WIN32)
  # TODO: compile on windows
else()
  message( FATAL_ERROR "Operating system not supported." )
endif()

# build fat binary for several common architectures
set(CUDA_NVCC_FLAGS "-O3 -std=c++11 -gencode=arch=compute_52,code=compute_52")

include_directories("include")

set(SOURCES
  "src/linop/block.cu"
  "src/linop/block_zero.cu"
  "src/linop/linearoperator.cu"

  "src/prox/prox.cu"
  "src/prox/prox_moreau.cu"
  "src/prox/prox_zero.cu"

#  "src/solver/backend_pdhg.cu"
#  "src/solver/problem.cu"
#  "src/solver/solver.cu"

  "include/exception.hpp"

  "include/linop/block.hpp"
  "include/linop/block_zero.hpp"
  "include/linop/linearoperator.hpp"

  "include/prox/prox.hpp"
  "include/prox/prox_moreau.hpp"
  "include/prox/prox_zero.hpp"

#  "include/solver/solver.hpp"
#  "include/solver/problem.hpp"
#  "include/solver/backend.hpp"
#  "include/solver/backend_pdhg.hpp"
  )
  
cuda_add_library(pdsolver STATIC ${SOURCES})
