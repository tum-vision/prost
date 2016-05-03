/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

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
