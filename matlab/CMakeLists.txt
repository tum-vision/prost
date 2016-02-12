set(CMAKE_CXX_COMPILER ${Matlab_mex})
set(CMAKE_C_COMPILER ${Matlab_mex})

if(MSVC)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .mexw64)
endif()

add_definitions(-DMATLAB_MEX_FILE)

set(CMAKE_CXX_FLAGS "-largeArrayDims")

if(UNIX)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .mexa64)
endif()

set(CMAKE_SHARED_LIBRARY_PREFIX)

set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")

set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS) # remove -shared options as mex does not accept it

if(APPLE)
  # this hack is necessary, as FindCUDA adds rpath under MacOSX and mex does not accept it
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> -lut -lcudart -lcusparse -lprost -L${CMAKE_BINARY_DIR}/src")
else()
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# TODO: make this more pretty
set(SOURCES
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/custom.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/factory.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/prost.cpp"
)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")

if(MSVC)
  include_directories("${Matlab_DIR}/extern/include")
  link_directories("${Matlab_DIR}/extern/lib/win64/microsoft/")
endif()

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(CMAKE_SHARED_LINKER_FLAGS "LDFLAGS='\$LDFLAGS -Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib'") 
endif()

add_library( prost_ SHARED ${SOURCES} )

if(MSVC)
  target_link_libraries( prost_ prost libmex libmx libut ${CUDA_cusparse_LIBRARY} )
  set_property(TARGET prost_ PROPERTY LINK_FLAGS "/export:mexFunction")
else()
  target_link_libraries( prost_ prost ut cusparse )
  add_dependencies( prost_ prost ) #required.
endif()

set_property(TARGET prost_ PROPERTY POSITION_INDEPENDENT_CODE False)

# MSVC adds Debug/Release directory, so install to correct location
if(MSVC)
  install(TARGETS prost_ DESTINATION "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
else()
  set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
endif()
