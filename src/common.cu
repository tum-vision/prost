#include "prost/common.hpp"

namespace prost {

#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)
  
string get_version() {
  return AS_STRING(PROST_VERSION);
}

template<typename T>
std::list<double> linspace(T start_in, T end_in, int num_in) {
  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);
  double delta = (end - start) / (num - 1);

  std::list<double> linspaced; 
  for(int i = 0; i < num; ++i) 
    linspaced.push_back(start + delta * i);

  linspaced.push_back(end);
  
  return linspaced;
}

template std::list<double> linspace<double>(double, double, int);
template std::list<double> linspace<float>(float, float, int);
template std::list<double> linspace<size_t>(size_t, size_t, int);
template std::list<double> linspace<int>(int, int, int);

// TODO: this is really awful C code. rewrite this.
template<typename T>
void csr2csc(
  int n, int m, int nz, 
  T *a, int *col_idx, int *row_start,
  T *csc_a, int *row_idx, int *col_start)
{
  int i, j, k, l;
  int *ptr;

  for (i=0; i<=m; i++) col_start[i] = 0;

  /* determine column lengths */
  for (i=0; i<nz; i++) col_start[col_idx[i]+1]++;
  for (i=0; i<m; i++) col_start[i+1] += col_start[i];

  /* go through the structure once more. Fill in output matrix. */
  for (i=0, ptr=row_start; i<n; i++, ptr++)
    for (j=*ptr; j<*(ptr+1); j++){
      k = col_idx[j];
      l = col_start[k]++;
      row_idx[l] = i;
      if (a) csc_a[l] = a[j];
    }

  /* shift back col_start */
  for (i=m; i>0; i--) col_start[i] = col_start[i-1];

  col_start[0] = 0;
}

// Beginning of GPU Architecture definitions                                                                                                                                                                                                  
int _ConvertSMVer2Cores(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM                                                                                                                                            
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version                                                                                                                                                
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
    {
      { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class                                                                                                                                                                                
      { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class                                                                                                                                                                                
      { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class                                                                                                                                                                               
      { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class                                                                                                                                                                               
      { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class                                                                                                                                                                               
      { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class                                                                                                                                                                               
      { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class                                                                                                                                                                              
      { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class                                                                                                                                                                              
      {   -1, -1 }
    };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly                                                                                                                                                           
  //printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
  return nGpuArchCoresPerSM[index-1].Cores;
}

template void csr2csc<float>(int n, int m, int nz, 
                             float *a, int *col_idx, int *row_start,
                             float *csc_a, int *row_idx, int *col_start);

template void csr2csc<double>(int n, int m, int nz, 
                              double *a, int *col_idx, int *row_start,
                              double *csc_a, int *row_idx, int *col_start);



} // namespace prost