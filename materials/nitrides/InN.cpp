#include "InN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

#include <cmath>

namespace plask {

std::string InN::name() const { return NAME; }

MI_PROPERTY(InN, condT,
            MISource("H. Tong et al., Proc. SPIE 7602 (2010) 76020U")
            )
double InN::condT(double T) const {
    return( 126. );
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
double InN::Me(double T, char point) const {
    double tMe(0.);
    if (point == 'G') tMe = 0.048;
    return (tMe);
}

MI_PROPERTY(InN, Me_v,
            MISource("Yan et al., Sem. Sci. Technol. 26 (2011) 014037"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double InN::Me_v(double T, char point) const {
    double tMe_v(0.);
    if (point == 'G') tMe_v = 0.068;
    return (tMe_v);
}

MI_PROPERTY(InN, Me_l,
            MISeeClass<InN>(MaterialInfo::Me_v)
            )
double InN::Me_l(double T, char point) const {
    double tMe_l(0.);
    if (point == 'G') tMe_l = 0.065;
    return (tMe_l);
}

MI_PROPERTY(InN, Mhh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
double InN::Mhh(double T, char point) const {
    double tMhh(0.);
    if (point == 'G') tMhh = 2.631;
    return (tMhh);
}

MI_PROPERTY(InN, Mhh_v,
            MISeeClass<InN>(MaterialInfo::Me_v)
            )
double InN::Mhh_v(double T, char point) const {
    double tMhh_v(0.);
    if (point == 'G') tMhh_v = 1.7007;
    return (tMhh_v);
}

MI_PROPERTY(InN, Mhh_l,
            MISeeClass<InN>(MaterialInfo::Me_v)
            )
double InN::Mhh_l(double T, char point) const {
    double tMhh_l(0.);
    if (point == 'G') tMhh_l = 1.8116;
    return (tMhh_l);
}

MI_PROPERTY(InN, Mlh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
double InN::Mlh(double T, char point) const {
    double tMlh(0.);
    if (point == 'G') tMlh = 2.631;
    return (tMlh);
}

MI_PROPERTY(InN, Mlh_v,
            MISeeClass<InN>(MaterialInfo::Me_v)
            )
double InN::Mlh_v(double T, char point) const {
    double tMlh_v(0.);
    if (point == 'G') tMlh_v = 0.0348;
    return (tMlh_v);
}

MI_PROPERTY(InN, Mlh_l,
            MISeeClass<InN>(MaterialInfo::Me_v)
            )
double InN::Mlh_l(double T, char point) const {
    double Mlh_l(0.);
    if (point == 'G') Mlh_l = 1.8116;
    return (Mlh_l);
}

MaterialsDB::Register<InN> materialDB_register_InN;

}       // namespace plask
