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

#ifndef PROX_H
#define PROX_H

#include "../../config.h"

class prox {
public:
  prox() : index(-1), count(-1), dim(-1), interleaved(false), diagprox(false) {}
  prox(int idx, int cnt, int d, bool inter, bool diag)
      : index(idx), count(cnt), dim(d), interleaved(inter), diagprox(diag) {}
  virtual ~prox() {}

  virtual void eval(real *d_proxarg,
                    real *d_result,
                    real tau,
                    real *d_tau,
                    bool invert_tau = false) = 0;
  
  int get_index() const { return index; }
  int get_dim() const { return dim; }
  int get_count() const { return count; }
  bool is_interleaved() const { return interleaved; }
  bool is_diagprox() const { return diagprox; }
  
protected:
  int index, count, dim;
  bool interleaved, diagprox;
};

#endif
