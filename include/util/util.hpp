#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <list>

template<typename T>
std::list<double> linspace(T start_in, T end_in, int num_in) {
  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);
  double delta = (end - start) / (num - 1);

  std::list<double> linspaced; 
  for(int i = 0; i < num; ++i) {
    linspaced.push_back(start + delta * i);
  }
  linspaced.push_back(end);
  
  return linspaced;
}

#endif
