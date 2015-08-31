/*
 * This file is part of pdsolver.
 *
 * Copyright (C) 2015 Thomas Möllenhoff <thomas.moellenhoff@in.tum.de> 
 *
 * pdsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pdsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pdsolver. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PROX_FUN_1D_H
#define PROX_FUN_1D_H

#include <algorithm>
#include <string>

enum prox_fun_1d {
  kZero = 0,       
  kAbs,            
  kSquare,
  kIndicatorLeq,
  kIndicatorEq,
  kIndicatorAbsLeq,

  kUnknownProx = -1
};

inline prox_fun_1d prox_fun_from_string(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  if("zero" == name) return kZero;
  else if("abs" == name) return kAbs;
  else if("square" == name) return kSquare;
  else if("leq" == name) return kIndicatorLeq;
  else if("eq" == name) return kIndicatorEq;
  else if("absleq" == name) return kIndicatorAbsLeq;

  return kUnknownProx;
}

#endif
