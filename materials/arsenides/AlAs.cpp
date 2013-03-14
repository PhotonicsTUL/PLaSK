#include "AlAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlAs::name() const { return NAME; }

MI_PROPERTY(AlAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.6611 + 2.90e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(AlAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double AlAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.099, 0.885e-3, 530., T);
    return ( tEg );
}

MI_PROPERTY(AlAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlAs::Dso(double T, double e) const {
    return ( 0.28 );
}

MI_PROPERTY(AlAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.124;
        tMe.c11 = 0.124;
    }
    return ( tMe );
}

MI_PROPERTY(AlAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs::Mhh(double T, double e, char point) const {
    Tensor2<double> tMhh(0.51, 0.51); // [001]
    return ( tMhh );
}

MI_PROPERTY(AlAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs::Mlh(double T, double e, char point) const {
    Tensor2<double> tMlh(0.18, 0.18);
    return ( tMlh );
}

MI_PROPERTY(AlAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlAs::ac(double T) const {
    return ( -5.64 );
}

MI_PROPERTY(AlAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlAs::av(double T) const {
    return ( 2.47 );
}

MI_PROPERTY(AlAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlAs::b(double T) const {
    return ( -2.3 );
}

MI_PROPERTY(AlAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlAs::c11(double T) const {
    return ( 125.0 );
}

MI_PROPERTY(AlAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double AlAs::c12(double T) const {
    return ( 53.4 );
}

MI_PROPERTY(AlAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // 300 K
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs::thermk(double T, double t) const {
    return(Tensor2<double>(91., 91.));
}

bool AlAs::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<AlAs> materialDB_register_AlAs;

}       // namespace plask
