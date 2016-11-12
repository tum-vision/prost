# prost
prost is a framework for solving large-scale problems with **pro**ximal **st**ructure. It contains efficient CUDA implementations of several primal-dual algorithms such as ADMM or PDHG and common proximal and linear operators.

The general class of problems that can be solved is:

![id](http://latex.codecogs.com/svg.latex?\\min_{x%20\\in\\mathbb{R}^n}%20%20\\max_{y%20\\in%20\\mathbb{R}^m}%20~%20g(x)%20+%20\\langle%20Kx,%20y%20\\rangle-%20f^*(y),)

where g and f* are convex functions mapping to the extended real line, whose proximal mapping is cheap to evaluate and K is a linear operator.

## Installation

#### Dependencies 

Make sure to have the newest [CUDA](https://developer.nvidia.com/cuda-downloads) toolkit installed and that nvcc is in the current path. We recommend to use a GPU of [compute capability](https://developer.nvidia.com/cuda-gpus) at least 3.0. 

The other dependency is [MATLAB](http://www.mathworks.com/). Interfaces to C/C++ and Python are planned for future releases.

#### Quick start

	git clone https://github.com/tum-vision/prost.git
	cd prost
	mkdir build
	cd build
	cmake ..
	make

## Getting started
To get familiar with the framework, we recommend looking at the MATLAB examples. To do so, start MATLAB and add the folder `/matlab/` to your path. Move to the folder `/matlab/examples/` and run any of the examples such as `example_rof_primaldual.m`.

To get an overview over the implemented proximal and linear operators look into the directories `/matlab/+prost/+function` and `/matlab/+prost/+block`. For more information about the individual functions you can use the help command within MATLAB, e.g.,

	help prost.function.sum_1d
	help prost.function.sum_norm2
	help prost.block.diags

## Troubleshooting / Hints
#### MacOSX
Tested using Apple LLVM version 7.0.2 (clang-700.1.81), CUDA 7.5 and matlab-R2015b. 

For CUDA 8.0, use Apple LLVM version 7.3.0 (clang-703.0.31), Xcode 7.3.1. See also the installation guide [here](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/#axzz4MVW0QgfH).

In case of the error message "No supported compiler or SDK was found.", 
one needs to add the line

 	<dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />
 
 to the clang and clang++ config files
 
	edit ([matlabroot '/bin/maci64/mexopts/clang_maci64.xml'])
 	edit ([matlabroot '/bin/maci64/mexopts/clang++_maci64.xml'])
	
Finally, run from within MATLAB

	mex -setup
	mex -setup C++

#### Linux
Tested using gcc-4.8, matlab-R2016a and CUDA 7.5, as well as gcc-5.4, matlab-R2016b and CUDA 8.0.

In case of an invalid mex-file error, start MATLAB with the following preload command:

	LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6" matlab-R2015b

#### Windows
Tested using Visual Studio 2013 Community (english version) and CUDA 7.5. Run `cmake-gui` and select `Visual Studio 12 2013 Win64` as generator. The following values might need to be set manually

- Set `CUDA_HOST_COMPILER` to the path where `cl.exe` is located
- Set `Matlab_DIR` and `Matlab_ROOT_DIR` to the MATLAB directory

Once these values have been set, run configure and generate once more
in cmake. Finally, open the Visual Studio solution and compile as
Release x64 and after compilation has finished build the project
INSTALL in the solution, which copies the mex file to the correct
directory.

## How to Cite
If you use prost in your scientific publication, please cite the paper

   **Sublabel-Accurate Relaxation of Nonconvex Energies**
     (T. Möllenhoff, E. Laude, M. Moeller, J. Lellmann, D. Cremers),
     In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
