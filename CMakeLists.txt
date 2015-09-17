cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
  set(CMAKE_CXX_FLAGS "-stdlib=libstdc++ -std=c++11 -O3")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "-fPIC -O3")
endif()

set(CUDA_NVCC_FLAGS "-arch=compute_30 -code=sm_30 -O3 --use_fast_math")
include_directories("include")
include_directories("include/util")

file(GLOB SOURCES
  "src/*.cpp"
  "src/*.cu"
  "include/*.hpp"
  "include/util/*.hpp"
  )
  
cuda_add_library(pdsolver STATIC ${SOURCES})
