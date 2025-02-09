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
#include "AlOx.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlOx::name() const { return NAME; }

MI_PROPERTY(AlOx, cond,
            MISource("A. Inoue et al., Journal of Materials Science 22 (1987) 2063-2068"),
            MINote("no temperature dependence")
            )
Tensor2<double> AlOx::cond(double /*T*/) const {
    return ( Tensor2<double>(1e-7, 1e-7) );
}

MI_PROPERTY(AlOx, thermk,
            MISource("M. Le Du et al., Electronics Letters 42 (2006) 65-66"),
            MINote("no temperature dependence")
            )
Tensor2<double> AlOx::thermk(double /*T*/, double /*h*/) const {
    return ( Tensor2<double>(0.7, 0.7) );
}

MI_PROPERTY(AlOx, absp,
            MISource(""),
            MINote("TODO")
            )
double AlOx::absp(double /*lam*/, double /*T*/) const {
    return ( 0. );
}

bool AlOx::isEqual(const Material &/*other*/) const {
    return true;
}

MI_PROPERTY(AlOx, nr,
            MISource("T.Kitatani et al., Japanese Journal of Applied Physics (part1) 41 (2002) 2954-2957"),
            MINote("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MINote("no temperature dependence"),
            MIArgumentRange(MaterialInfo::lam, 400, 1600)
			)
double AlOx::nr(double lam, double /*T*/, double /*n*/) const {
    return ( 0.30985*exp(-lam/236.7)+1.52829 );
}

static MaterialsDB::Register<AlOx> materialDB_register_AlOx;

}}       // namespace plask::materials
