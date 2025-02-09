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
#include "AuGe.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AuGe::name() const { return NAME; }

MI_PROPERTY(AuGe, cond,
            MISource("T. Som, P. Ayyub, D. Kabiraj, N. Kulkarni, V.N. Kulkarni and D.K. Avasthi, Formation of Au0.6Ge0.4 alloy induced by Au-ion irradiation of Au/Ge bilayer, Journal of Applied Physics, vol. 84, no. 2, pp. 3861-3863, 2004."),
            MINote("no temperature dependence")
            )
Tensor2<double> AuGe::cond(double /*T*/) const {
    double tCond = 1e8;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AuGe, thermk,
            MISource("www.thinfilm.com"),
            MINote("no temperature dependence")
            )
Tensor2<double> AuGe::thermk(double /*T*/, double /*t*/) const {
    double tCondT = 88.34;
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool AuGe::isEqual(const Material &/*other*/) const {
    return true;
}

// MI_PROPERTY(AuGe, absp,
//             MISource(""),
//             MINote("TODO")
//             )
// double AuGe::absp(double /*lam*/, double /*T*/) const {
//     return ( 1e3 );
// }
//
// MI_PROPERTY(AuGe, nr,
//             MISource(""),
//             MINote("TODO")
//             )
// double AuGe::nr(double /*lam*/, double /*T*/, double /*n*/) const {
//     return ( 1. );
// }

static MaterialsDB::Register<AuGe> materialDB_register_AuGe;

}}       // namespace plask::materials
