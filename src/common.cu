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

template void csr2csc<float>(int n, int m, int nz, 
                             float *a, int *col_idx, int *row_start,
                             float *csc_a, int *row_idx, int *col_start);

template void csr2csc<double>(int n, int m, int nz, 
                              double *a, int *col_idx, int *row_start,
                              double *csc_a, int *row_idx, int *col_start);

} // namespace prost