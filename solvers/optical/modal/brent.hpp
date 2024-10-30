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
#ifndef PLASK__OPTICAL_EFFECTIVE_BRENT_H
#define PLASK__OPTICAL_EFFECTIVE_BRENT_H

#include "rootdigger.hpp"
#include "solver.hpp"

namespace plask { namespace optical { namespace modal {

class RootBrent: public RootDigger {

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chartName(); prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  protected:

    double axisBrent(dcomplex start, double& fx, bool real, int& counter);

  public:

    // Constructor
    RootBrent(ModalBase& solver, const function_type& val_fun, const Params& pars, const char* name):
        RootDigger(solver, val_fun, pars, name) {}


    dcomplex find(dcomplex start) override;
};

}}} // namespace plask::optical::modal
#endif // PLASK__OPTICAL_EFFECTIVE_BRENT_H
