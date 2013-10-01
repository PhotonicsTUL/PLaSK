#include "InAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAs::name() const { return NAME; }

MI_PROPERTY(InAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.0583 + 2.74e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(InAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double InAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.417, 0.276e-3, 93., T);
    return ( tEg );
}

MI_PROPERTY(InAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::Dso(double T, double e) const {
    return ( 0.39 );
}

MI_PROPERTY(InAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.024;
        tMe.c11 = 0.024;
    }
    return ( tMe );
}

MI_PROPERTY(InAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAs::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.26, 0.26); // [001]
    return ( tMhh );
}

MI_PROPERTY(InAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAs::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.027, 0.027);
    return ( tMlh );
}

MI_PROPERTY(InAs, VBO,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::VBO(double T, double e, char point) const {
    return ( -0.59 );
}

MI_PROPERTY(InAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::ac(double T) const {
    return ( -5.08 );
}

MI_PROPERTY(InAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::av(double T) const {
    return ( 1.00 );
}

MI_PROPERTY(InAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::b(double T) const {
    return ( -1.8 );
}

MI_PROPERTY(InAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::c11(double T) const {
    return ( 83.29 );
}

MI_PROPERTY(InAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double InAs::c12(double T) const {
    return ( 45.26 );
}

MI_PROPERTY(InAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MISource("W. Nakwaski, J. Appl. Phys. 64 (1988) 159"),
            MIArgumentRange(MaterialInfo::T, 300, 650)
            )
Tensor2<double> InAs::thermk(double T, double t) const {
    double tCondT = 30.*pow((300./T),1.234);
    return(Tensor2<double>(tCondT, tCondT));
}

bool InAs::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<InAs> materialDB_register_InAs;

}} // namespace plask::materials
