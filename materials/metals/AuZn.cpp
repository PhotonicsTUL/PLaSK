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
#include "AuZn.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AuZn::name() const { return NAME; }

MI_PROPERTY(AuZn, cond,
            MISource("C. Belouet, C. Villard, C. Fages and D. Keller, Achievement of homogeneous AuSn solder by pulsed laser-assisted deposition, Journal of Electronic Materials, vol. 28, no. 10, pp. 1123-1126, 1999."),
            MINote("no temperature dependence")
            )
Tensor2<double> AuZn::cond(double /*T*/) const {
    double tCond = 1e6; // TODO (check this value: AuZn or AuSn)
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AuZn, thermk,
            MISource("D. Singh and D.K. Pandey, Ultrasonic investigations in intermetallics, Pramana - Journal of Physics, vol. 72, no. 2, pp. 389-398, 2009."),
            MINote("no temperature dependence")
            )
Tensor2<double> AuZn::thermk(double /*T*/, double /*t*/) const {
    double tCondT = 110.3;
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool AuZn::isEqual(const Material &/*other*/) const {
    return true;
}

// MI_PROPERTY(AuZn, absp,
//             MISource(""),
//             MINote("TODO")
//             )
// double AuZn::absp(double /*lam*/, double /*T*/) const {
//     return ( 1e3 );
// }
//
// MI_PROPERTY(AuZn, nr,
//             MISource(""),
//             MINote("TODO")
//             )
// double AuZn::nr(double /*lam*/, double /*T*/, double /*n*/) const {
//     return ( 1. );
// }

static MaterialsDB::Register<AuZn> materialDB_register_AuZn;

}}       // namespace plask::materials
