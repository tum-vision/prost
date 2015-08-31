cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER /usr/bin/clang)
set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
file(GLOB SRC_SOLVER
  "pdsolver/*.cpp"
  "pdsolver/*.cu"
  "pdsolver/*.h"
  "pdsolver/backend/*.cpp"
  "pdsolver/backend/*.cu"
  "pdsolver/backend/*.h"
  "pdsolver/prox/*.cpp"
  "pdsolver/prox/*.cu"
  "pdsolver/prox/*.h"
  )
  
cuda_add_library(pdsolver STATIC ${SRC_SOLVER})


