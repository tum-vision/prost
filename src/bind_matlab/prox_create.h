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

#ifndef PROX_CREATE_H
#define PROX_CREATE_H

#include "mex.h"
#include <vector>

class prox;

bool prox_from_matlab(const mxArray *pm, std::vector<prox *>& result);

#endif
