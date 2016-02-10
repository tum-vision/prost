# Broken on mac. Add rpath somewhere?

find_package(MatlabMex REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER ${Matlab_mex})
set(CMAKE_C_COMPILER ${Matlab_mex})

# TODO is this even necessary?
#set(CMAKE_SHARED_LIBRARY_SUFFIX .mexa64)
#set(CMAKE_SHARED_LIBRARY_PREFIX)

set(CMAKE_CXX_FLAGS "-largeArrayDims")

set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> <DEFINES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")

set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS) # remove -shared options as mex does not accept it
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")

set(SOURCES
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/custom.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/factory.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/prost.cpp"
)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CUDA_TOOLKIT_ROOT_DIR}/lib64")
#set(CMAKE_SKIP_RPATH TRUE)

#include_directories(${CUDA_INCLUDE_DIRS})
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
add_library( prost_ SHARED ${SOURCES} )
target_link_libraries( prost_ prost ut cusparse )
set_property(TARGET prost_ PROPERTY POSITION_INDEPENDENT_CODE False)

# # Caffe-style: is this the better alternative?
# # TODO: do this properly
# set(Matlab_prost_mex ${CMAKE_SOURCE_DIR}/matlab/+prost/mex/prost_)

# set(Matlab_srcs
#   "${CMAKE_SOURCE_DIR}/matlab/+prost/mex/custom.cpp"
#   "${CMAKE_SOURCE_DIR}/matlab/+prost/mex/factory.cpp"
#   "${CMAKE_SOURCE_DIR}/matlab/+prost/mex/prost.cpp"
#   "${CMAKE_SOURCE_DIR}/build/src/libprost.a"
# )

# set(Matlab_cxxflags CXXFLAGS='\\$CXXFLAGS -std=c++11')
# set(Matlab_ldflags LDFLAGS='\\$LDFLAGS -Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib64')
# set(Matlab_includepath -I${CMAKE_SOURCE_DIR}/include/ -I${CUDA_TOOLKIT_ROOT_DIR}/include)
# set(Matlab_libpath -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -L${CMAKE_SOURCE_DIR}/build/src)
# set(Matlab_linkerflags -lcusparse -lcublas -lcudart -lut )

# set(Matlab_args -largeArrayDims -output ${Matlab_prost_mex} ${Matlab_cxxflags} ${Matlab_ldflags} ${Matlab_linkpath} ${Matlab_includepath} ${Matlab_libpath} ${Matlab_srcs} ${Matlab_linkerflags})

# add_custom_command(OUTPUT ${Matlab_prost_mex} COMMAND ${Matlab_mex} ARGS ${Matlab_args} DEPENDS prost COMMENT "Building Matlab interface: ${Matlab_prost_mex}" VERBATIM)

# add_custom_target(matlab ALL DEPENDS ${Matlab_prost_mex} SOURCES ${Matlab_srcs})
