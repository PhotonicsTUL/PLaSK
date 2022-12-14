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
#ifndef PLASK__OPTICAL_EFFECTIVE_BROYDEN_H
#define PLASK__OPTICAL_EFFECTIVE_BROYDEN_H

#include "rootdigger.hpp"

namespace plask { namespace optical { namespace effective {

class RootBroyden: public RootDigger {

    // Parameters for Broyden algorithm
    static constexpr double EPS = 1e6 * SMALL; ///< precision for computing Jacobian

    // Return Jacobian of F(x)
    void fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji) const;

    // Search for the new point x along direction p for which f decreased sufficiently
    bool lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax) const;

    // Search for the root of char_val using globally convergent Broyden method
    dcomplex Broyden(dcomplex x) const;

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chartName(); prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  public:

    // Constructor
    RootBroyden(Solver& solver, const function_type& val_fun, DataLog<dcomplex,dcomplex>& log_value,
               const Params& pars): RootDigger(solver, val_fun, log_value, pars) {}

    dcomplex find(dcomplex start) const override;
};

}}} // namespace plask::optical::effective
#endif // PLASK__OPTICAL_EFFECTIVE_BROYDEN_H
