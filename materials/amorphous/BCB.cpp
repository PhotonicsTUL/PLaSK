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
#include "BCB.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string BCB::name() const { return NAME; }

MI_PROPERTY(BCB, cond,
            MISource("The DOW Chemical Company, CYCLOTENE Advanced Electronic Resins (2005) 1-9"),
            MINote("no temperature dependence")
            )
Tensor2<double> BCB::cond(double /*T*/) const {
    return ( Tensor2<double>(1e-17, 1e-17) );
}

MI_PROPERTY(BCB, thermk,
            MISource("X. Xu et al., IEEE Components, Packaging, and Manufacturing Technology 2 (2012) 286-293"),
            MINote("fit for pure BCB by Lukasz Piskorski, unpublished"),
            MIArgumentRange(MaterialInfo::T, 290, 420)
            )
Tensor2<double> BCB::thermk(double T, double /*h*/) const {
    double tK = 0.31*pow(300./T,-1.1); // [tK] = W/(m*K)
    return ( Tensor2<double>(tK, tK) );
}

Material::ConductivityType BCB::condtype() const { return Material::CONDUCTIVITY_OTHER; }

MI_PROPERTY(BCB, dens,
            MISource("A. Modafe et al., Microelectronic Engineering 82 (2005) 154-167")
            )
double BCB::dens(double /*T*/) const {
    return ( 1050. ); // kg/m^3
}

MI_PROPERTY(BCB, cp,
            MISource("A. Modafe et al., Microelectronic Engineering 82 (2005) 154-167")
            )
double BCB::cp(double /*T*/) const {
    return ( 2180. ); // J/(kg*K)
}

bool BCB::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<BCB> materialDB_register_BCB;

}}       // namespace plask::materials
