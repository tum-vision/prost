cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
  set(CMAKE_CXX_FLAGS "-O3 -stdlib=libstdc++ -std=c++11")
else()
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -std=c++11")
endif()

# TODO: fix this
#if(UNIX)
#endif()

set(CUDA_NVCC_FLAGS "-arch=sm_52 -O3 -std=c++11")
#set(CUDA_NVCC_FLAGS "-arch=compute_12 -code=sm_12 -O3 -std=c++11")

include_directories("include")

file(GLOB SOURCES
  "src/*.cpp"
  "src/*.cu"
  "src/prox/*.cpp"
  "src/prox/*.cu"
  "src/prox/elemop/*.cpp"
  "src/prox/elemop/*.cu"
  "include/*.hpp"
  "include/prox/*.hpp"
  "include/prox/elemop/*.hpp"
  "include/util/*.hpp"
  )
  
cuda_add_library(pdsolver STATIC ${SOURCES})
