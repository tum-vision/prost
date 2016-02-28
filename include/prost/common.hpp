#ifndef PROST_COMMON_HPP_
#define PROST_COMMON_HPP_

#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

namespace prost {

// common functions from standard library
using std::cout;
using std::endl;
using std::function;
using std::list;
using std::map;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

string get_version(); 
template<typename T> list<double> linspace(T start_in, T end_in, int num_in);

/// \brief Helper function that converts CSR format to CSC format, 
///        not in-place, if a == NULL, only pattern is reorganized
///        the size of matrix is n x m.
template<typename T>
void csr2csc(int n, int m, int nz, 
             T *a, int *col_idx, int *row_start,
             T *csc_a, int *row_idx, int *col_start); 

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

} // namespace prost

#endif // PROST_COMMON_HPP_
