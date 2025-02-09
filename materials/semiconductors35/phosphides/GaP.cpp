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
#include "GaP.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaP::name() const { return NAME; }

MI_PROPERTY(GaP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.4505 + 2.92e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(GaP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(2.896, 0.96e-3, 423., T);
    else if (point == 'X') tEg = phys::Varshni(2.35, 0.5771e-3, 372., T);
    else if (point == 'L') tEg = phys::Varshni(2.72, 0.5771e-3, 372., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(2.896, 0.96e-3, 423., T);
        double tEgX = phys::Varshni(2.35, 0.5771e-3, 372., T);
        double tEgL = phys::Varshni(2.72, 0.5771e-3, 372., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::Dso(double /*T*/, double /*e*/) const {
    return ( 0.08 );
}

MI_PROPERTY(GaP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MINote("no temperature dependence")
            )
Tensor2<double> GaP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.114), tMeX(1.58), tMeL(0.75);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == 'L') {
        tMe.c00 = tMeL; tMe.c11 = tMeL;
    }
    else if (point == '*') {
        double EgG = Eg(T,e,'G'), EgX = Eg(T,e,'X'), EgL = Eg(T,e,'L');
        if (EgG <= EgX && EgG <= EgL) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        } else if (EgX <= EgL){
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        } else {
            tMe.c00 = tMeL; tMe.c11 = tMeL;
        }
    } else
        throw Exception("GaP: Me: bad point '{c}'", point);
    return ( tMe );
}

MI_PROPERTY(GaP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
            )
Tensor2<double> GaP::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.34, 0.34); // [001]
}

MI_PROPERTY(GaP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
            )
Tensor2<double> GaP::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.20, 0.20);
}

MI_PROPERTY(GaP, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence"),
            MINote("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaP::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-1.27);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(GaP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::ac(double /*T*/) const {
    return ( -8.2 );
}

MI_PROPERTY(GaP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::av(double /*T*/) const {
    return ( 1.7 );
}

MI_PROPERTY(GaP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::b(double /*T*/) const {
    return ( -1.6 );
}

MI_PROPERTY(GaP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::d(double /*T*/) const {
    return ( -4.6 );
}

MI_PROPERTY(GaP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::c11(double /*T*/) const {
    return ( 140.5 );
}

MI_PROPERTY(GaP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::c12(double /*T*/) const {
    return ( 62.03 );
}

MI_PROPERTY(GaP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
            )
double GaP::c44(double /*T*/) const {
    return ( 70.33 );
}

MI_PROPERTY(GaP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67"), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 60, 535)
            )
Tensor2<double> GaP::thermk(double T, double /*t*/) const {
    double tCondT = 77.*pow((300./T),1.35);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(GaP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MINote("no temperature dependence")
            )
double GaP::dens(double /*T*/) const { return 4.1299e3; }

MI_PROPERTY(GaP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MINote("no temperature dependence")
            )
double GaP::cp(double /*T*/) const { return 0.313e3; }

Material::ConductivityType GaP::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaP, nr,
            MINote("TODO")
            )
double GaP::nr(double /*lam*/, double /*T*/, double /*n*/) const {
    throw NotImplemented("nr for GaP");
}

MI_PROPERTY(GaP, absp,
            MINote("TODO")
            )
double GaP::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for GaP");
}

bool GaP::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<GaP> materialDB_register_GaP;

}} // namespace plask::materials
