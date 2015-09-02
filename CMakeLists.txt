cmake_minimum_required(VERSION 2.6)

project(pdsolver)

find_package(CUDA REQUIRED)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
  set(CMAKE_CXX_FLAGS "-stdlib=libstdc++ -std=c++11 -g")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "-fPIC -g")
endif()

set(CUDA_NVCC_FLAGS "-arch=compute_30 -code=sm_30 --use_fast_math -O3")
include_directories("include")
include_directories("include/util")

file(GLOB SOURCES
  "src/*.cpp"
  "src/*.cu"
  "include/*.hpp"
  "include/util/*.hpp"
  )
  
cuda_add_library(pdsolver STATIC ${SOURCES})
