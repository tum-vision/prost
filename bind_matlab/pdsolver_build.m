if ismac
    cuda_lib = '/usr/local/cuda/lib';
    cuda_inc = '/usr/local/cuda/include';
elseif isunix
    cuda_lib = '/work/sdks/cudacurrent/lib64';
    cuda_inc = '/work/sdks/cudacurrent/include';
elseif ispc
    % TODO: set cuda paths
else
    disp('Cannot recognize platform');
    return;
end

unix('make -C ../build/ -j8');

if ismac
    eval(sprintf(['mex -largeArrayDims -output pdsolver ' ...
                  'CXXFLAGS=''\\$CXXFLAGS -stdlib=libstdc++'' '...
                  'LDFLAGS=''\\$LDFLAGS -stdlib=libstdc++ -Wl,-rpath,%s'' '...
                  'mex/mex_pdsolver.cpp ../build/libpdsolver.a' ...
                  ' -L%s -I%s -lcudart -lcublas -lcusparse -lut -I../include' ], ...
                 cuda_lib, cuda_lib, cuda_inc))
    
elseif isunix
    eval(sprintf(['mex -largeArrayDims -output pdsolver ' ...
                  'CXXFLAGS=''\\$CXXFLAGS -std=c++11'' '...
                  'LDFLAGS=''\\$LDFLAGS -static-libstdc++ -Wl,-rpath,%s'' '...
                  'mex/mex_pdsolver.cpp ../build/libpdsolver.a' ...
                  ' -L%s -I%s -lcudart -lcublas -lcusparse -lut -I../include' ], ...
                  cuda_lib, cuda_lib, cuda_inc))
elseif ispc
    % TODO: build on windows
end

