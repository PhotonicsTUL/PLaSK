#include "GaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaAs::name() const { return NAME; }

MI_PROPERTY(GaAs, cond,
            MISource("-"),
			MIComment("fit to n-GaAs:Si (based on 8 papers 1982 - 2003)"),
			MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs::cond(double T) const {
    return ( std::make_pair(940., 940.) ); // TODO
}

MI_PROPERTY(GaAs, thermCond,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MISource("W. Nakwaski, J. Appl. Phys. 64 (1988) 159"),
            MIArgumentRange(MaterialInfo::T, 300, 900)
            )
std::pair<double,double> GaAs::thermCond(double T, double t) const {
    double tCondT = 45.*pow((300./T),1.25);
    return ( std::make_pair(tCondT, tCondT) );
}

MI_PROPERTY(GaAs, absp,
            MISource(""),
            MIComment("TODO")
            )
double GaAs::absp(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaAs, nr,
            MISource(""),
            MIComment("TODO")
            )
double GaAs::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815–5875")
            )
double GaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.65325 + 3.88e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(GaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815–5875"),
            MIComment("only for Gamma point")
            )
double GaAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 1.519 - 0.5405e-3 * T * T / (T + 204.); //  cFunc::Varshni(1.519, 0.5405e-3, 204., T);
    return ( tEg );
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

static MaterialsDB::Register<GaAs> materialDB_register_GaAs;

}       // namespace plask
