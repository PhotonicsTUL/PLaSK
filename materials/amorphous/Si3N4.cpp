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
#include "Si3N4.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Si3N4::name() const { return NAME; }

MI_PROPERTY(Si3N4, cond,
            MISource(""),
            MINote("TODO")
            )
Tensor2<double> Si3N4::cond(double /*T*/) const {
    throw NotImplemented("cond for Si3N4");
}

MI_PROPERTY(Si3N4, thermk,
            MISource(""),
            MINote("TODO")
            )
Tensor2<double> Si3N4::thermk(double /*T*/, double /*h*/) const {
    throw NotImplemented("thermk for Si3N4");
}

Material::ConductivityType Si3N4::condtype() const { return Material::CONDUCTIVITY_OTHER; }

MI_PROPERTY(Si3N4, nr,
            MISource("refractiveindex.info"),
            MIArgumentRange(MaterialInfo::lam, 207, 1240)
            )
double Si3N4::nr(double lam, double /*T*/, double /*n*/) const {
    double tL2 = lam*lam*1e-6;
    return ( sqrt(1+2.8939*tL2/(tL2-0.0195077089)));
}
MI_PROPERTY(Si3N4, absp,
            MISource("S. Zhou et al., Proc. SPIE 7995 (2011) 79950T"),
            MINote("data for SiNx"),
            MIArgumentRange(MaterialInfo::lam, 9000, 11000)
            )
double Si3N4::absp(double lam, double /*T*/) const {
    double tL = lam*1e-3;
    return ( 1.06E-4*pow(tL,7.8) );
}
bool Si3N4::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<Si3N4> materialDB_register_Si3N4;

}}       // namespace plask::materials
