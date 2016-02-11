#ifndef PROST_EXCEPTION_HPP_
#define PROST_EXCEPTION_HPP_

#include <stdexcept>
#include <string>

namespace prost {

class Exception : public std::exception {
public:
  explicit Exception(const char *message) : msg_(message) {}
  explicit Exception(const std::string& message) : msg_(message) {}
  virtual ~Exception() throw () {}

  virtual const char* what() const throw() {
    return msg_.c_str();
  }

protected:
  std::string msg_;
};

} // namespace prost

#endif // PROST_EXCEPTION_HPP_
