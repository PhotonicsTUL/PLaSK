#include "InAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InAs::name() const { return NAME; }

MI_PROPERTY(InAs, condT,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MISource("W. Nakwaski, J. Appl. Phys. 64 (1988) 159"),
            MIArgumentRange(MaterialInfo::T, 300, 650)
            )
std::pair<double,double> InAs::condT(double T, double t) const {
    double tCondT = 30*pow((T/300.),-1.234);
    return(std::make_pair(tCondT, tCondT));
 }

MI_PROPERTY(InAs, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double InAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.0583;
    return (tLattC);
}

MI_PROPERTY(InAs, Eg,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double InAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 0.359;
    return (tEg);
}

MI_PROPERTY(InAs, Me,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> InAs::Me(double T, char point) const {
    std::pair<double,double> tMe(0., 0.);
    if (point == 'G') {
        tMe.first = 0.024;
        tMe.second = 0.024;
    }
    return (tMe);
}

MI_PROPERTY(InAs, Mhh,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> InAs::Mhh(double T, char point) const {
    std::pair<double,double> tMhh(0., 0.);
    if (point == 'G') {
        tMhh.first = 0.26;
        tMhh.second = 0.45;
    }
    return (tMhh);
}

MI_PROPERTY(InAs, Mlh,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> InAs::Mlh(double T, char point) const {
    std::pair<double,double> tMlh(0., 0.);
    if (point == 'G') {
        tMlh.first = 0.027;
        tMlh.second = 0.026;
    }
    return (tMlh);
}

static MaterialsDB::Register<InAs> materialDB_register_InAs;

}       // namespace plask
