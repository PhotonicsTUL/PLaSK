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
#include "rootdigger.hpp"
#include "solverbase.hpp"

namespace plask { namespace optical { namespace modal {

RootDigger::RootDigger(ModalBase& solver, const function_type& val_fun, const Params& pars, const char* name) :
    solver(solver),
    val_function(val_fun),
    params(pars),
    log_value(solver.getId(), "modal", name, "det")
{}

}}} // namespace plask::optical::modal
