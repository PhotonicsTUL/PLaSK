#include "AlP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlP::name() const { return NAME; }

MI_PROPERTY(AlP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.4672 + 2.92e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(AlP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.63, 0.5771e-3, 372., T);
    else if (point == 'X') tEg = phys::Varshni(2.52, 0.318e-3, 588., T);
    else if (point == 'L') tEg = phys::Varshni(3.57, 0.318e-3, 588., T);
    return ( tEg );
}

MI_PROPERTY(AlP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::Dso(double T, double e) const {
    return ( 0.07 );
}

MI_PROPERTY(AlP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.220;
        tMe.c11 = 0.220;
    }
    return ( tMe );
}

MI_PROPERTY(AlP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlP::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.30, 0.30); // [001]
    return ( tMhh );
}

MI_PROPERTY(AlP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlP::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.28, 0.28);
    return ( tMlh );
}

MI_PROPERTY(AlP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::VB(double T, double e, char point) const {
    double tVB(-1.74);
    if (!e) return ( tVB );
    else return ( tVB + 2.*av(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::ac(double T) const {
    return ( -5.7 );
}

MI_PROPERTY(AlP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::av(double T) const {
    return ( 3.0 );
}

MI_PROPERTY(AlP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::b(double T) const {
    return ( -1.5 );
}

MI_PROPERTY(AlP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::d(double T) const {
    return ( -4.6 );
}

MI_PROPERTY(AlP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::c11(double T) const {
    return ( 133.0 );
}

MI_PROPERTY(AlP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::c12(double T) const {
    return ( 63.0 );
}

MI_PROPERTY(AlP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlP::c44(double T) const {
    return ( 61.5 );
}

MI_PROPERTY(AlP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // k(300K)
            MIComment("temperature dependence assumed similar to be similar to k for AlSb")
            )
Tensor2<double> AlP::thermk(double T, double t) const {
    double tCondT = (1./0.011)*pow((300./T),1.4);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(AlP, nr,
            MISource(""),
            MIComment("TODO")
            )
double AlP::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlP, absp,
            MISource("TODO"),
            MIComment("TODO")
            )
double AlP::absp(double wl, double T) const {
    return ( 0. );
}

bool AlP::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<AlP> materialDB_register_AlP;

}} // namespace plask::materials
