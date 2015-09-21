#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <ctime>
#include <list>
#include <string>

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

// measuring time
class Timer
{
 public:
  Timer() : tStart(0), running(false), sec(0.f)
  {
  }
  void start()
  {
    tStart = clock();
    running = true;
  }
  void end()
  {
    if (!running) { sec = 0; return; }
    cudaDeviceSynchronize();
    clock_t tEnd = clock();
    sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
    running = false;
  }
  float get()
  {
    if (running) end();
    return sec;
  }
 private:
  clock_t tStart;
  bool running;
  float sec;
};

#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

#endif
