/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER2D_PYTHON_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER2D_PYTHON_H

#include "fourier-python.hpp"
#include "../fourier/solver2d.hpp"

namespace plask { namespace optical { namespace slab { namespace python {

void export_FourierSolver2D();

}}}} // # namespace plask::optical::slab::python

#endif // PLASK__SOLVER__OPTICAL__SLAB_FOURIER2D_PYTHON_H
