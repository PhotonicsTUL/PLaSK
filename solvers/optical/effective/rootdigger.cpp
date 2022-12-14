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
#include "muller.hpp"
#include "broyden.hpp"
#include "brent.hpp"

namespace plask { namespace optical { namespace effective {

std::unique_ptr<RootDigger> RootDigger::get(Solver* solver, const function_type& func, DataLog<dcomplex,dcomplex>& detlog, const Params& params) {
    typedef std::unique_ptr<RootDigger> Res;
    if (params.method == RootDigger::ROOT_MULLER) return Res(new RootMuller(*solver, func, detlog, params));
    else if (params.method == RootDigger::ROOT_BROYDEN) return Res(new RootBroyden(*solver, func, detlog, params));
    else if (params.method == RootDigger::ROOT_BRENT) return Res(new RootBrent(*solver, func, detlog, params));
    throw BadInput(solver->getId(), "Wrong root finding method");
    return Res();
}

void RootDigger::readRootDiggerConfig(XMLReader& reader, Params& params) {
    params.tolx = reader.getAttribute<double>("tolx", params.tolx);
    params.tolf_min = reader.getAttribute<double>("tolf-min", params.tolf_min);
    params.tolf_max = reader.getAttribute<double>("tolf-max", params.tolf_max);
    params.maxstep = reader.getAttribute<double>("maxstep", params.maxstep);
    params.maxiter = reader.getAttribute<int>("maxiter", params.maxiter);
    params.alpha = reader.getAttribute<double>("alpha", params.alpha);
    params.lambda_min = reader.getAttribute<double>("lambd", params.lambda_min);
    params.initial_dist = reader.getAttribute<dcomplex>("initial-range", params.initial_dist);
    params.method = reader.enumAttribute<RootDigger::Method>("method")
        .value("brent", RootDigger::ROOT_BRENT)
        .value("broyden", RootDigger::ROOT_BROYDEN)
        .value("muller", RootDigger::ROOT_MULLER)
        .get(params.method);
    params.stairs = reader.getAttribute<int>("stairs", params.stairs);
    reader.requireTagEnd();
}

}}} // namespace plask::optical::effective
