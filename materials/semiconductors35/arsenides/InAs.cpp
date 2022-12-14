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
#include "InAs.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PROPERTY(InAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double InAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.0583 + 2.74e-5 * (T-300.);
    return tLattC;
}

MI_PROPERTY(InAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double InAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.417, 0.276e-3, 93., T);
    else if (point == 'X') tEg = phys::Varshni(1.433, 0.276e-3, 93., T);
    else if (point == 'L') tEg = phys::Varshni(1.133, 0.276e-3, 93., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(0.417, 0.276e-3, 93., T);
        double tEgX = phys::Varshni(1.433, 0.276e-3, 93., T);
        double tEgL = phys::Varshni(1.133, 0.276e-3, 93., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::Dso(double /*T*/, double /*e*/) const {
    return 0.39;
}

MI_PROPERTY(InAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MINote("no temperature dependence")
           )
Tensor2<double> InAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.024), tMeX(0.98), tMeL(0.94);
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
        throw Exception("InAs: Me: bad point '{c}'", point);
    return ( tMe );
}

MI_PROPERTY(InAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
           )
Tensor2<double> InAs::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.26, 0.26); // [001]
}

MI_PROPERTY(InAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
           )
Tensor2<double> InAs::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.027, 0.027);
}

MI_PROPERTY(InAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence"),
            MINote("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> InAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(InAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double InAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}

MI_PROPERTY(InAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-0.59);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(InAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::ac(double /*T*/) const {
    return -5.08;
}

MI_PROPERTY(InAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::av(double /*T*/) const {
    return 1.00;
}

MI_PROPERTY(InAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::b(double /*T*/) const {
    return -1.8;
}

MI_PROPERTY(InAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::d(double /*T*/) const {
    return -3.6;
}

MI_PROPERTY(InAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::c11(double /*T*/) const {
    return 83.29;
}

MI_PROPERTY(InAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::c12(double /*T*/) const {
    return 45.26;
}

MI_PROPERTY(InAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double InAs::c44(double /*T*/) const {
    return 39.59;
}

MI_PROPERTY(InAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67"), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 300, 650)
           )
Tensor2<double> InAs::thermk(double T, double /*t*/) const {
    double tCondT = 30.*pow((300./T),1.73);
    return(Tensor2<double>(tCondT, tCondT));
}

MI_PROPERTY(InAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MINote("no temperature dependence")
            )
double InAs::dens(double /*T*/) const { return 5.6678e3; }

MI_PROPERTY(InAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MINote("no temperature dependence")
            )
double InAs::cp(double /*T*/) const { return 0.352e3; }

Material::ConductivityType InAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(InAs, eps,
            MISource("http://www.iue.tuwien.ac.at/phd/quay/node27.html")
)
double InAs::eps(double /*T*/) const {
    return 14.6;
}

std::string InAs::name() const { return NAME; }

bool InAs::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<InAs> materialDB_register_InAs;

}} // namespace plask::materials
