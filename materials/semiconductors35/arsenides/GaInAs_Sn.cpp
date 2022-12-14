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
#include "GaInAs_Sn.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaInAs_Sn::name() const { return NAME; }

std::string GaInAs_Sn::str() const { return StringBuilder("In", In)("Ga")("As").dopant("Sn", ND); }

MI_PARENT(GaInAs_Sn, GaInAs)

GaInAs_Sn::GaInAs_Sn(const Material::Composition& Comp, double Val): GaInAs(Comp)/*, mGaAs_Sn(Type,Val), mInAs_Sn(Type,Val)*/
{
    Nf_RT = Val; // TODO
    ND = Val; // TODO

    if (In == 0.53)
        mob_RT = 10600./(1+pow((Nf_RT/2e17),0.39));
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_Sn, mob,
            MISource("TODO"),
            MISource("based on Si-doped GaInAs")
            )
Tensor2<double> GaInAs_Sn::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_Sn, Nf,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
double GaInAs_Sn::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double GaInAs_Sn::doping() const {
    return ( ND );
}

MI_PROPERTY(GaInAs_Sn, cond,
            MINote("no temperature dependence")
            )
Tensor2<double> GaInAs_Sn::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType GaInAs_Sn::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(GaInAs_Sn, absp,
            MISource("fit to ..."), // TODO
            MINote("no temperature dependence")
            )
double GaInAs_Sn::absp(double lam, double /*T*/) const {
    double tAbsp(0.);
    if ((lam > 1200.) && (lam < 1400.)) // only for 1300 nm TODO
        tAbsp = 18600. * pow(Nf_RT/1e18-3.1, -0.64);
    else if ((lam > 1450.) && (lam < 1650.)) // only for 1550 nm TODO
        tAbsp = 7600. * pow(Nf_RT/1e18, -2.0);
    return ( tAbsp );
}

bool GaInAs_Sn::isEqual(const Material &other) const {
    const GaInAs_Sn& o = static_cast<const GaInAs_Sn&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaInAs::isEqual(other);
}

static MaterialsDB::Register<GaInAs_Sn> materialDB_register_GaInAs_Sn;

}} // namespace plask::materials
