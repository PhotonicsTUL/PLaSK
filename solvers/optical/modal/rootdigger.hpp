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
#ifndef PLASK__OPTICAL_SLAB_ROOTDIGGER_H
#define PLASK__OPTICAL_SLAB_ROOTDIGGER_H

#include <functional>
#include <plask/plask.hpp>

namespace plask { namespace optical { namespace modal {

struct ModalBase;

struct RootDigger {

    typedef std::function<dcomplex(dcomplex)> function_type;

    /// Root finding method
    enum Method {
        ROOT_MULLER,
        ROOT_BROYDEN,
        ROOT_BRENT
    };

    struct Params {
        Method method;          ///< Root finding method
        double tolx,            ///< Absolute tolerance on the argument
            tolf_min,           ///< Sufficient tolerance on the function value
            tolf_max,           ///< Required tolerance on the function value
            maxstep;            ///< Maximum step in one iteration
        int maxiter;            ///< Maximum number of iterations
        double alpha,           ///< Ensures sufficient decrease of determinant in each step
            lambda_min;         ///< Minimum decrease ratio of the step (lambda)
        dcomplex initial_dist;  ///< Distance between initial points

        Params():
            method(ROOT_MULLER),
            tolx(1e-6),
            tolf_min(1e-7),
            tolf_max(1e-5),
            maxstep(0.1),
            maxiter(500),
            alpha(1e-7),
            lambda_min(1e-8),
            initial_dist(1e-3) {}
    };

  protected:

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const;

    // Solver
    ModalBase& solver;

    // Solver method computing the value to zero
    function_type val_function;

    // Value writelog
    DataLog<dcomplex,dcomplex> log_value;

    inline dcomplex valFunction(dcomplex x) const {
        try {
            return val_function(x);
        } catch (...) {
            log_value.throwError(x);
        }
        return NAN;
    }

  public:

    // Rootdigger parameters
    Params params;

    // Constructor
    RootDigger(ModalBase& solver, const function_type& val_fun, const Params& pars, const char* name);

    virtual ~RootDigger() {}

    /**
     * Search for a single zero starting from the given point
     * \param start initial point to start search from
     * \return found solution
     */
    virtual dcomplex find(dcomplex start) = 0;
};

}}} // namespace plask::optical::modal

#endif // PLASK__OPTICAL_SLAB_ROOTDIGGER_H
