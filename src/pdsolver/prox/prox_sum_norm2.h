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

#ifndef PROX_SUM_NORM2_H
#define PROX_SUM_NORM2_H

#include "prox_sum_1d.h"

class prox_sum_norm2 : public prox_sum_1d {
public:
  prox_sum_norm2(int idx, int cnt, int dim, bool interleaved,
                 const prox_1d_coefficients& prox_coeffs,
                 const prox_fun_1d& fn);
  virtual ~prox_sum_norm2();

  virtual void eval(real *d_proxarg,
                    real *d_result,
                    real tau,
                    real *d_tau,
                    bool invert_tau = false);
};

#endif
