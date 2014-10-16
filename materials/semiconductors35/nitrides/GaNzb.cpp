#include "GaNzb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

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
    else if (point == '*')
    {
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
            MIComment("no temperature dependence")
           )
double GaNzb::Dso(double T, double e) const {
    return 0.017;
}


MI_PROPERTY(GaNzb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaNzb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.15;
        tMe.c11 = 0.15;
    }
    return tMe;
}


MI_PROPERTY(GaNzb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaNzb::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.83, 0.83); // [001]
    return tMhh;
}


MI_PROPERTY(GaNzb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaNzb::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.22, 0.22);
    return tMlh;
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
            MIComment("no temperature dependence")
           )
double GaNzb::VB(double T, double e, char point, char hole) const {
    double tVB(-2.64);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
    return tVB;
}


MI_PROPERTY(GaNzb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::ac(double T) const {
    return -6.71;
}


MI_PROPERTY(GaNzb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::av(double T) const {
    return 0.69;
}


MI_PROPERTY(GaNzb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::b(double T) const {
    return -2.0;
}


MI_PROPERTY(GaNzb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::d(double T) const {
    return -3.7;
}


MI_PROPERTY(GaNzb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::c11(double T) const {
    return 293.;
}


MI_PROPERTY(GaNzb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::c12(double T) const {
    return 159.;
}


MI_PROPERTY(GaNzb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 94 (2003) 3675-3696"),
            MIComment("no temperature dependence")
           )
double GaNzb::c44(double T) const {
    return 155.;
}


MI_PROPERTY(GaNzb, thermk,
            MISource(""),
            MIComment("TODO")
           )
Tensor2<double> GaNzb::thermk(double T, double t) const {
    return Tensor2<double>(0., 0.);
}


MI_PROPERTY(GaNzb, cond,
            MISource(""),
            MIComment("TODO")
           )
Tensor2<double> GaNzb::cond(double T) const {
    return 0.;
}


MI_PROPERTY(GaNzb, nr,
            MISource(""),
            MIComment("TODO")
           )
double GaNzb::nr(double wl, double T, double n) const {
    return 0.;
}


MI_PROPERTY(GaNzb, absp,
            MISource(""),
            MIComment("TODO")
           )
double GaNzb::absp(double wl, double T) const {
    return 0.;
}


MI_PROPERTY(GaNzb, eps,
            MISource(""),
            MIComment("TODO")
           )
double GaNzb::eps(double T) const {
    return 0.;
}

std::string GaNzb::name() const { return NAME; }

bool GaNzb::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<GaNzb> materialDB_register_GaNzb;

}} // namespace plask::materials
