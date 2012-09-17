#include "AlAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlAs::name() const { return NAME; }

MI_PROPERTY(AlAs, condT,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlAs::condT(double T, double t) const {
    return(std::make_pair(91., 91.));
 }

MI_PROPERTY(AlAs, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double AlAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.6614;
    return (tLattC);
}

MI_PROPERTY(AlAs, Eg,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double AlAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 3.01;
    return (tEg);
}

MI_PROPERTY(AlAs, Me,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlAs::Me(double T, char point) const {
    std::pair<double,double> tMe(0., 0.);
    if (point == 'G') {
        tMe.first = 0.124;
        tMe.second = 0.124;
    }
    return (tMe);
}

MI_PROPERTY(AlAs, Mhh,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlAs::Mhh(double T, char point) const {
    std::pair<double,double> tMhh(0., 0.);
    if (point == 'G') {
        tMhh.first = 0.51;
        tMhh.second = 1.09;
    }
    return (tMhh);
}

MI_PROPERTY(AlAs, Mlh,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlAs::Mlh(double T, char point) const {
    std::pair<double,double> tMlh(0., 0.);
    if (point == 'G') {
        tMlh.first = 0.18;
        tMlh.second = 0.15;
    }
    return (tMlh);
}

static MaterialsDB::Register<AlAs> materialDB_register_AlAs;

}       // namespace plask
