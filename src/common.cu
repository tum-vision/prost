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

} // namespace prost