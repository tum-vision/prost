#include "prost/common.hpp"

namespace prost {

#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)
  
string get_version() {
  return AS_STRING(PROST_VERSION);
}

} // namespace prost