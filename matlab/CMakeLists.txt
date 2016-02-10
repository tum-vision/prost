find_package(MatlabMex REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER ${Matlab_mex})
set(CMAKE_C_COMPILER ${Matlab_mex})

if(MSVC)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .mexw64)
  add_definitions(-DMATLAB_MEX_FILE)
endif()

if(APPLE)
  # TODO
endif()

# TODO: Broken on mac. Add rpath somewhere? -- Test this!
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CUDA_TOOLKIT_ROOT_DIR}/lib64")
#set(CMAKE_SKIP_RPATH TRUE)

if(UNIX)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .mexa64)
endif()

set(CMAKE_SHARED_LIBRARY_PREFIX)

# TODO: Could this cause any problems for large matrices on Windows?
if(APPLE OR UNIX)
  set(CMAKE_CXX_FLAGS "-largeArrayDims")
endif()

set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> <DEFINES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")

set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS) # remove -shared options as mex does not accept it
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")

# TODO: make this more pretty
set(SOURCES
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/custom.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/factory.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/prost.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/factory.hpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/config.hpp"
)

if(MSVC)
  include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
  include_directories("${Matlab_DIR}/extern/include")
  link_directories("${Matlab_DIR}/extern/lib/win64/microsoft/")
endif()

add_library( prost_ SHARED ${SOURCES} )

if(MSVC)
  target_link_libraries( prost_ prost libmex libmx libut ${CUDA_cusparse_LIBRARY} )
  set_property(TARGET prost_ PROPERTY LINK_FLAGS "/export:mexFunction")
else()
  target_link_libraries( prost_ prost ut cusparse )
endif()

set_property(TARGET prost_ PROPERTY POSITION_INDEPENDENT_CODE False)

# MSVC adds Debug/Release directory, so install to correct location
if(MSVC)
  install(TARGETS prost_ DESTINATION "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
else()
  set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
endif()