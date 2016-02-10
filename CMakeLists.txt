cmake_minimum_required(VERSION 2.8.7)

project(prost C CXX)

set(PROST_TARGET_VERSION "0.0.1")
add_definitions(-DPROST_VERSION=${PROST_TARGET_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()	

include_directories("include")

add_subdirectory(src)
add_subdirectory(matlab)
