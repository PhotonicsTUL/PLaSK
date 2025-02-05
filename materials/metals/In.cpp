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
#include "In.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string In::name() const { return NAME; }

MI_PROPERTY(In, cond,
            MISource("www.thinfilm.com"),
            MINote("no temperature dependence")
            )
Tensor2<double> In::cond(double /*T*/) const {
    const double tCond = 1.392e7;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(In, thermk,
            MISource("www.lakeshore.com"),
            MINote("no temperature dependence")
            )
Tensor2<double> In::thermk(double /*T*/, double /*t*/) const {
    const double tCondT = 84.;
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool In::isEqual(const Material &/*other*/) const {
    return true;
}

// MI_PROPERTY(In, absp,
//             MISource(""),
//             MINote("TODO")
//             )
// double In::absp(double /*lam*/, double /*T*/) const {
//     return ( 1e3 );
// }
//
// MI_PROPERTY(In, nr,
//             MISource(""),
//             MINote("TODO")
//             )
// double In::nr(double /*lam*/, double /*T*/, double /*n*/) const {
//     return ( 1. );
// }

static MaterialsDB::Register<In> materialDB_register_In;

}}       // namespace plask::materials
