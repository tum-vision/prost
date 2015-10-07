cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
  set(CMAKE_CXX_FLAGS "-O3 -stdlib=libstdc++")
else()
  set(CMAKE_CXX_FLAGS "-fPIC -O3")
endif()

# TODO: fix this
#if(UNIX)
#endif()

set(CUDA_NVCC_FLAGS "-O3 -arch=sm_52")
#set(CUDA_NVCC_FLAGS "-arch=compute_12 -code=sm_12 -O3")

include_directories("include")

file(GLOB SOURCES
  "src/*.cpp"
  "src/*.cu"
  "src/linop/*.cpp"
  "src/linop/*.cu"
  "src/prox/*.cpp"
  "src/prox/*.cu"
  "src/solver/*.cpp"
  "src/solver/*.cu"
  "src/util/*.cpp"
  "src/util/*.cu"
  "include/*.hpp"
  "include/linop/*.hpp"
  "include/prox/*.hpp"
  "include/solver/*.hpp"
  "include/util/*.hpp"
  )
  
cuda_add_library(pdsolver STATIC ${SOURCES})
