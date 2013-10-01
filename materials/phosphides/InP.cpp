#include "InP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InP::name() const { return NAME; }

MI_PROPERTY(InP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.8697 + 2.79e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(InP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double InP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(1.4236, 0.363e-3, 162., T);
    return ( tEg );
}

MI_PROPERTY(InP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::Dso(double T, double e) const {
    return ( 0.108 );
}

MI_PROPERTY(InP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.07927;
        tMe.c11 = 0.07927;
    }
    return ( tMe );
}

MI_PROPERTY(InP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.46, 0.46); // [001]
    return ( tMhh );
}

MI_PROPERTY(InP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.12, 0.12);
    return ( tMlh );
}

MI_PROPERTY(InP, VBO,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::VBO(double T, double e, char point) const {
    return ( -0.94 );
}

MI_PROPERTY(InP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::ac(double T) const {
    return ( -6.0 );
}

MI_PROPERTY(InP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::av(double T) const {
    return ( 0.6 );
}

MI_PROPERTY(InP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::b(double T) const {
    return ( -2.0 );
}

MI_PROPERTY(InP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::c11(double T) const {
    return ( 101.1 );
}

MI_PROPERTY(InP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InP::c12(double T) const {
    return ( 56.1 );
}

MI_PROPERTY(InP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // k(300K)
            MISource("I. Kudman et al., Phys. Rev. 133 (1964) A1665-A1667"), // experimental data k(T)
            MISource("L. Piskorski, unpublished"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 300, 800)
            )
Tensor2<double> InP::thermk(double T, double t) const {
    double tCondT = 68.*pow((300./T),1.5);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(InP, nr,
            MISource(""),
            MIComment("TODO")
            )
double InP::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(InP, absp,
            MISource("TODO"),
            MIComment("TODO")
            )
double InP::absp(double wl, double T) const {
    return ( 0. );
}

bool InP::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<InP> materialDB_register_InP;

}} // namespace plask::materials
