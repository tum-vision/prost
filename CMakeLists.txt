cmake_minimum_required(VERSION 2.8.7)

project(prost C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/CustomSources.cmake")
  include(cmake/CustomSources.cmake)
endif()

if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267")
endif()

find_package(MatlabMex REQUIRED)
find_package(CUDA REQUIRED)

include_directories("include")
	
add_subdirectory(src)
add_subdirectory(matlab)
