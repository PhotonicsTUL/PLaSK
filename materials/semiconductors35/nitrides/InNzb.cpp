#include "InNzb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {


MI_PROPERTY(InNzb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double InNzb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 4.98 + 2.83e-5 * (T-300.);
    return tLattC;
}


MI_PROPERTY(InNzb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double InNzb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.78, 0.245e-3, 624., T);
    else if (point == 'X') tEg = phys::Varshni(2.51, 0.245e-3, 624., T);
    else if (point == 'L') tEg = phys::Varshni(5.82, 0.245e-3, 624., T);
    return tEg;
}


MI_PROPERTY(InNzb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::Dso(double T, double e) const {
    return 0.005;
}


MI_PROPERTY(InNzb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
           )
Tensor2<double> InNzb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.03;
        tMe.c11 = 0.03;
    }
    return tMe;
}


MI_PROPERTY(InNzb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
           )
Tensor2<double> InNzb::Mhh(double T, double e) const {
    Tensor2<double> tMhh(1.261, 1.261); // [001]
    return tMhh;
}


MI_PROPERTY(InNzb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
           )
Tensor2<double> InNzb::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.100, 0.100);
    return tMlh;
}


MI_PROPERTY(InNzb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696")
           )
double InNzb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}


MI_PROPERTY(InNzb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::VB(double T, double e, char point, char hole) const {
    double tVB(-2.34);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
    return tVB;
}


MI_PROPERTY(InNzb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::ac(double T) const {
    return -2.65;
}


MI_PROPERTY(InNzb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::av(double T) const {
    return 0.7;
}


MI_PROPERTY(InNzb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::b(double T) const {
    return -1.2;
}


MI_PROPERTY(InNzb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::d(double T) const {
    return -9.3;
}


MI_PROPERTY(InNzb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::c11(double T) const {
    return 187.;
}


MI_PROPERTY(InNzb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::c12(double T) const {
    return 125.;
}


MI_PROPERTY(InNzb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double InNzb::c44(double T) const {
    return 86.;
}


MI_PROPERTY(InNzb, thermk,
            MISource(""),
            MIComment("TODO")
           )
Tensor2<double> InNzb::thermk(double T, double t) const {
    return Tensor2<double>(0., 0.);
}


MI_PROPERTY(InNzb, cond,
            MISource(""),
            MIComment("TODO")
           )
Tensor2<double> InNzb::cond(double T) const {
    return 0.;
}


MI_PROPERTY(InNzb, nr,
            MISource(""),
            MIComment("TODO")
           )
double InNzb::nr(double wl, double T, double n) const {
    return 0.;
}


MI_PROPERTY(InNzb, absp,
            MISource(""),
            MIComment("TODO")
           )
double InNzb::absp(double wl, double T) const {
    return 0.;
}


MI_PROPERTY(InNzb, eps,
            MISource(""),
            MIComment("TODO")
           )
double InNzb::eps(double T) const {
    return 0.;
}

std::string InNzb::name() const { return NAME; }

bool InNzb::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<InNzb> materialDB_register_InNzb;

}} // namespace plask::materials
