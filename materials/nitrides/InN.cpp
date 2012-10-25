#include "InN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

#include <cmath>

namespace plask {

std::string InN::name() const { return NAME; }

MI_PROPERTY(InN, thermk,
            MISource("H. Tong et al., Proc. SPIE 7602 (2010) 76020U")
            )
std::pair<double,double> InN::thermk(double T) const {
    return(std::make_pair(126., 126.));
 }

MI_PROPERTY(InN, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double InN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 3.548;
    else if (x == 'c') tLattC = 5.760;
    return (tLattC);
}

MI_PROPERTY(InN, Eg,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double InN::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 0.77;
    return (tEg);
}

MI_PROPERTY(InN, Me,
            MISource("King et al., Phys. Rev. B 75 (2007) 115312"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> InN::Me(double T, char point) const {
    std::pair<double,double> tMe(0.,0.);
    if (point == 'G') {
        tMe.first = 0.065;
        tMe.second = 0.068;
    }
    return (tMe);
}

MI_PROPERTY(InN, Mhh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
std::pair<double,double> InN::Mhh(double T, char point) const {
    std::pair<double,double> tMhh(0.,0.);
    if (point == 'G') {
        tMhh.first = 1.8116;
        tMhh.second = 1.7007;
    }
    return (tMhh);
}

MI_PROPERTY(InN, Mlh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
std::pair<double,double> InN::Mlh(double T, char point) const {
    std::pair<double,double> tMlh(0.,0.);
    if (point == 'G') {
        tMlh.first = 1.8116;
        tMlh.second = 0.0348;
    }
    return (tMlh);
}

MaterialsDB::Register<InN> materialDB_register_InN;

}       // namespace plask
