find_package(MatlabMex REQUIRED)

set(CMAKE_CXX_COMPILER /usr/local/lehrstuhl/DIR/matlab-R2015b/bin/mex) #${Matlab_mex})
set(CMAKE_C_COMPILER /usr/local/lehrstuhl/DIR/matlab-R2015b/bin/mex) #${Matlab_mex})

set(CMAKE_SHARED_LIBRARY_SUFFIX .mexa64)
set(CMAKE_SHARED_LIBRARY_PREFIX)

set(CMAKE_CXX_FLAGS "-cxx -largeArrayDims")
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)

set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> <DEFINES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")

set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS) 
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")

set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/matlab/+prost/mex/")

set(SOURCES 
  "+prost/mex/custom.cpp"
  "+prost/mex/factory.cpp"
  "+prost/mex/prost.cpp"

  "+prost/mex/config.hpp"
  "+prost/mex/factory.hpp"
)

add_library( prost_ SHARED ${SOURCES} )
target_link_libraries( prost_ prost ut cusparse )
