#include "GaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaAs::name() const { return NAME; }

MI_PROPERTY(GaAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.65325 + 3.88e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(GaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double GaAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(1.519, 0.5405e-3, 204., T);
    return ( tEg );
}

MI_PROPERTY(GaAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaAs::Dso(double T) const {
    return ( 0.341 );
}

MI_PROPERTY(GaAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs::Me(double T, char point) const {
    std::pair<double,double> tMe(0., 0.);
    if (point == 'G') {
        tMe.first = 0.067;
        tMe.second = 0.067;
    }
    return ( tMe );
}

MI_PROPERTY(GaAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs::Mhh(double T, char point) const {
    std::pair<double,double> tMhh(0.33, 0.33); // [001]
    return ( tMhh );
}

MI_PROPERTY(GaAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs::Mlh(double T, char point) const {
    std::pair<double,double> tMlh(0.090, 0.090);
    return ( tMlh );
}

MI_PROPERTY(GaAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaAs::ac(double T) const {
    return ( -7.17 );
}

MI_PROPERTY(GaAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaAs::av(double T) const {
    return ( 1.16 );
}

MI_PROPERTY(GaAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaAs::b(double T) const {
    return ( -2.0 );
}

MI_PROPERTY(GaAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaAs::c11(double T) const {
    return ( 122.1 );
}

MI_PROPERTY(GaAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaAs::c12(double T) const {
    return ( 56.6 );
}

MI_PROPERTY(GaAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // 300 K
            MISource("W. Nakwaski, J. Appl. Phys. 64 (1988) 159"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 300, 900)
           )
std::pair<double,double> GaAs::thermk(double T, double t) const {
    double tCondT = 45.*pow((300./T),1.25);
    return ( std::make_pair(tCondT, tCondT) );
}

MI_PROPERTY(GaAs, cond,
            MISource("http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/electric.html"),
            MIComment("Carrier concentration estimated")
           )
std::pair<double,double> GaAs::cond(double T) const {
    double c = 1e2 * phys::qe * 8000.* pow((300./T), 2./3.) * 1e16;
    return std::make_pair(c, c);
}


MI_PROPERTY(GaAs, nr,
            MISource(""),
            MIComment("TODO")
            )
double GaAs::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaAs, absp,
            MISource(""),
            MIComment("TODO")
            )
double GaAs::absp(double wl, double T) const {
    return ( 0. );
}

static MaterialsDB::Register<GaAs> materialDB_register_GaAs;

} // namespace plask
