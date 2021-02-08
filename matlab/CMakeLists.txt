if(UNIX)
  if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
      OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
    #    list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)

#    message(STATUS ${GCC_MAJOR})
#    message(STATUS ${GCC_MINOR})
  endif()
endif()

set(CMAKE_CXX_COMPILER ${Matlab_mex})
set(CMAKE_C_COMPILER ${Matlab_mex})

if(MSVC)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .mexw64)
endif()

add_definitions(-DMATLAB_MEX_FILE)

if(MSVC)
  # is largeArrayDims needed? since cuSparse cannot handle 64-bit indices anyways.
else()
  set(CMAKE_CXX_FLAGS "-largeArrayDims")
endif()

if(UNIX)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .mexa64)
endif()

set(CMAKE_SHARED_LIBRARY_PREFIX)

if(APPLE OR MSVC)
  set(CMAKE_CXX_COMPILE_OBJECT
    "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")
else()
  if(${GCC_MAJOR} GREATER 4)
    set(CMAKE_CXX_COMPILE_OBJECT
      "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")
  else()
    #somehow on GCC 4.x we have to remove <INCLUDES>
    set(CMAKE_CXX_COMPILE_OBJECT
      "<CMAKE_CXX_COMPILER> <DEFINES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp).o <OBJECT>")
  endif()

endif()

set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS) # remove -shared options as mex does not accept it

if(APPLE)
  # this hack is necessary, as FindCUDA adds rpath under MacOSX and mex does not accept it
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> -lut -lcudart -lcusparse -lcusolver -lcublas -lprost -L${CMAKE_BINARY_DIR}/src")
elseif(UNIX)
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> -lprost -lcublas -lcusparse -lcusolver -lcudart -lut -L${CMAKE_BINARY_DIR}/src")
else()
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

set(SOURCES
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/factory.cpp"
  "${CMAKE_SOURCE_DIR}/matlab/+prost/private/prost.cpp"
)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/matlab/+prost/private")

if(MSVC)
  include_directories("${Matlab_DIR}/extern/include")
  link_directories("${Matlab_DIR}/extern/lib/win64/microsoft/")
endif()

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(CMAKE_SHARED_LINKER_FLAGS "LDFLAGS='\$LDFLAGS -Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib' -L/usr/local/cuda/lib") 
endif()

add_library( prost_ SHARED ${SOURCES} ${MATLAB_CUSTOM_SOURCES})

if(MSVC)
  target_link_libraries( prost_ prost libmex libmx libut ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
  set_property(TARGET prost_ PROPERTY LINK_FLAGS "/export:mexFunction")
  set_property(TARGET prost_ PROPERTY  _CRT_SECURE_NO_WARNINGS )
else()
  target_link_libraries( prost_ prost libut ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} ) #cusparse cublas )
  add_dependencies( prost_ prost ) #required.
endif()

set_property(TARGET prost_ PROPERTY POSITION_INDEPENDENT_CODE False)

# MSVC adds Debug/Release directory, so install to correct location
if(MSVC)
  install(TARGETS prost_ DESTINATION "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
else()
  set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/matlab/+prost/private")
endif()
