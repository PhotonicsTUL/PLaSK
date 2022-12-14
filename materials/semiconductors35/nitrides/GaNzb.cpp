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
#include "GaNzb.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PROPERTY(GaNzb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double GaNzb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 4.50 + 2.05e-5 * (T-300.);
    return tLattC;
}

MI_PROPERTY(GaNzb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double GaNzb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.299, 0.593e-3, 600., T);
    else if (point == 'X') tEg = phys::Varshni(4.52, 0.593e-3, 600., T);
    else if (point == 'L') tEg = phys::Varshni(5.59, 0.593e-3, 600., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(3.299, 0.593e-3, 600., T);
        double tEgX = phys::Varshni(4.52, 0.593e-3, 600., T);
        double tEgL = phys::Varshni(5.59, 0.593e-3, 600., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaNzb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::Dso(double /*T*/, double /*e*/) const {
    return 0.017;
}

MI_PROPERTY(GaNzb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MINote("only for Gamma and X points"),
            MINote("no temperature dependence")
           )
Tensor2<double> GaNzb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.15), tMeX(0.78);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == '*') {
        double EgG = Eg(T,e,'G'), EgX = Eg(T,e,'X');
        if (EgG <= EgX) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        } else {
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        }
    } else
        throw Exception("GaNzb: Me: bad point '{c}'", point);
    return ( tMe );
}

MI_PROPERTY(GaNzb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
           )
Tensor2<double> GaNzb::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.83, 0.83); // [001]
}

MI_PROPERTY(GaNzb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
           )
Tensor2<double> GaNzb::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.22, 0.22);
}

MI_PROPERTY(GaNzb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence"),
            MINote("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaNzb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaNzb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double GaNzb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}

MI_PROPERTY(GaNzb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-2.64);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(GaNzb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::ac(double /*T*/) const {
    return -6.71;
}

MI_PROPERTY(GaNzb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::av(double /*T*/) const {
    return 0.69;
}

MI_PROPERTY(GaNzb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::b(double /*T*/) const {
    return -2.0;
}

MI_PROPERTY(GaNzb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::d(double /*T*/) const {
    return -3.7;
}

MI_PROPERTY(GaNzb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::c11(double /*T*/) const {
    return 293.;
}

MI_PROPERTY(GaNzb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::c12(double /*T*/) const {
    return 159.;
}

MI_PROPERTY(GaNzb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MINote("no temperature dependence")
           )
double GaNzb::c44(double /*T*/) const {
    return 155.;
}

MI_PROPERTY(GaNzb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MINote("no temperature dependence")
            )
double GaNzb::dens(double /*T*/) const { return 6.02e3; }

std::string GaNzb::name() const { return NAME; }

bool GaNzb::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<GaNzb> materialDB_register_GaNzb;

}} // namespace plask::materials
