#include "GaP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

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
    return ( tEg );
}

MI_PROPERTY(GaP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::Dso(double T, double e) const {
    return ( 0.08 );
}

MI_PROPERTY(GaP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.114;
        tMe.c11 = 0.114;
    }
    return ( tMe );
}

MI_PROPERTY(GaP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaP::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.34, 0.34); // [001]
    return ( tMhh );
}

MI_PROPERTY(GaP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaP::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.20, 0.20);
    return ( tMlh );
}

MI_PROPERTY(GaP, CBO,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaP::CBO(double T, double e, char point) const {
    double tCBO( VBO(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCBO );
    else return ( tCBO + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaP, VBO,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::VBO(double T, double e, char point) const {
    double tVBO(-1.27);
    if (!e) return ( tVBO );
    else return ( tVBO + 2.*av(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::ac(double T) const {
    return ( -8.2 );
}

MI_PROPERTY(GaP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::av(double T) const {
    return ( 1.7 );
}

MI_PROPERTY(GaP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::b(double T) const {
    return ( -1.6 );
}

MI_PROPERTY(GaP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::d(double T) const {
    return ( -4.6 );
}

MI_PROPERTY(GaP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::c11(double T) const {
    return ( 140.5 );
}

MI_PROPERTY(GaP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::c12(double T) const {
    return ( 62.03 );
}

MI_PROPERTY(GaP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaP::c44(double T) const {
    return ( 70.33 );
}

MI_PROPERTY(GaP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // k(300K)
            MISource("W. Nakwaski, J. Appl. Phys. 64 (1988) 159"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 300, 500)
            )
Tensor2<double> GaP::thermk(double T, double t) const {
    double tCondT = 77.*pow((300./T),1.364);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(GaP, nr,
            MISource(""),
            MIComment("TODO")
            )
double GaP::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaP, absp,
            MISource("TODO"),
            MIComment("TODO")
            )
double GaP::absp(double wl, double T) const {
    return ( 0. );
}

bool GaP::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<GaP> materialDB_register_GaP;

}} // namespace plask::materials
