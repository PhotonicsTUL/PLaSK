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
#include "AlInAs.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlInAs::AlInAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    In = Comp.find("In")->second;
}

std::string AlInAs::str() const { return StringBuilder("Al", Al)("In")("As"); }

std::string AlInAs::name() const { return NAME; }

Material::Composition AlInAs::composition() const {
    return { {"Al", Al}, {"In", In}, {"As", 1} };
}

MI_PROPERTY(AlInAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs")
            )
double AlInAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlAs.lattC(T,'a') + In*mInAs.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlAs.lattC(T,'c') + In*mInAs.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlInAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("nonlinear interpolation: AlAs, InAs")
            )
double AlInAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Al*In*(0.70);
    else if (point == 'X') tEg = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point);
    else if (point == 'L') tEg = Al*mAlAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point);
    else if (point == '*') {
        double tEgG = Al*mAlAs.Eg(T,e,'G') + In*mInAs.Eg(T,e,'G') - Al*In*(0.70);
        double tEgX = Al*mAlAs.Eg(T,e,'X') + In*mInAs.Eg(T,e,'X');
        double tEgL = Al*mAlAs.Eg(T,e,'L') + In*mInAs.Eg(T,e,'L');
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlInAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("nonlinear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::Dso(double T, double e) const {
    return ( Al*mAlAs.Dso(T,e) + In*mInAs.Dso(T,e) - Al*In*(0.15) );
}

MI_PROPERTY(AlInAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
Tensor2<double> AlInAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Al*mAlAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00,
        tMe.c11 = Al*mAlAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Al*mAlAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00;
        tMe.c11 = Al*mAlAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -Al*In*(0.012) );
        tMe.c11 += ( -Al*In*(0.012) );
    }
    return ( tMe );
}

MI_PROPERTY(AlInAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
Tensor2<double> AlInAs::Mhh(double T, double e) const {
    double lMhh = Al*mAlAs.Mhh(T,e).c00 + In*mInAs.Mhh(T,e).c00,
           vMhh = Al*mAlAs.Mhh(T,e).c11 + In*mInAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlInAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
Tensor2<double> AlInAs::Mlh(double T, double e) const {
    double lMlh = Al*mAlAs.Mlh(T,e).c00 + In*mInAs.Mlh(T,e).c00,
           vMlh = Al*mAlAs.Mlh(T,e).c11 + In*mInAs.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlInAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence"),
            MINote("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlInAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlInAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlInAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlInAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlAs.VB(T,0.,point,hole) + In*mInAs.VB(T,0.,point,hole) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(AlInAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::ac(double T) const {
    return ( Al*mAlAs.ac(T) + In*mInAs.ac(T) );
}

MI_PROPERTY(AlInAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::av(double T) const {
    return ( Al*mAlAs.av(T) + In*mInAs.av(T) );
}

MI_PROPERTY(AlInAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::b(double T) const {
    return ( Al*mAlAs.b(T) + In*mInAs.b(T) );
}

MI_PROPERTY(AlInAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::d(double T) const {
    return ( Al*mAlAs.d(T) + In*mInAs.d(T) );
}

MI_PROPERTY(AlInAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::c11(double T) const {
    return ( Al*mAlAs.c11(T) + In*mInAs.c11(T) );
}

MI_PROPERTY(AlInAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::c12(double T) const {
    return ( Al*mAlAs.c12(T) + In*mInAs.c12(T) );
}

MI_PROPERTY(AlInAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::c44(double T) const {
    return ( Al*mAlAs.c44(T) + In*mInAs.c44(T) );
}

MI_PROPERTY(AlInAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67"), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence for binaries
            MINote("inversion of nonlinear interpolation of resistivity: AlAs, InAs")
            )
Tensor2<double> AlInAs::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermk(T,t).c00 + In/mInAs.thermk(T,t).c00 + Al*In*0.15),
           vCondT = 1./(Al/mAlAs.thermk(T,t).c11 + In/mInAs.thermk(T,t).c11 + Al*In*0.15);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlInAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::dens(double T) const {
    return ( Al*mAlAs.dens(T) + In*mInAs.dens(T) );
}

MI_PROPERTY(AlInAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MINote("linear interpolation: AlAs, InAs"),
            MINote("no temperature dependence")
            )
double AlInAs::cp(double T) const {
    return ( Al*mAlAs.cp(T) + In*mInAs.cp(T) );
}

Material::ConductivityType AlInAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlInAs, nr,
            MISource("M.J. Mondry et al., IEEE Photon. Technol. Lett. 4 (1992) 627-630"),
            MINote("data for the wavelength ranging 1000-2000 nm")
            )
double AlInAs::nr(double lam, double T, double /*n*/) const {
    double tnr = sqrt(8.677 + (1.214*lam*lam)/(lam*lam-730.8*730.8)), //lam: 1000-2000 nm
           tBeta = 3.5e-4; //D. Dey for In = 0.365
    return ( tnr + tBeta*(T-300.) );
}

MI_PROPERTY(AlInAs, absp,
            MINote("TODO")
            )
double AlInAs::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for AlGaAs");
}

bool AlInAs::isEqual(const Material &other) const {
    const AlInAs& o = static_cast<const AlInAs&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlInAs> materialDB_register_AlInAs;

}} // namespace plask::materials
