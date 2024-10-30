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
#ifndef PLASK__OPTICAL_SLAB_MULLER_H
#define PLASK__OPTICAL_SLAB_MULLER_H

#include "rootdigger.hpp"
#include "solver.hpp"

namespace plask { namespace optical { namespace modal {

struct RootMuller: public RootDigger {

    RootMuller(ModalBase& solver, const function_type& val_fun, const Params& pars, const char* name):
        RootDigger(solver, val_fun, pars, name) {}


    dcomplex find(dcomplex start) override;
};

}}} // namespace plask::optical::modal
#endif // PLASK__OPTICAL_SLAB_MULLER_H
