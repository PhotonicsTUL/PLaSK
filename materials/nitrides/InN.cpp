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
Tensor2<double> InN::thermk(double T) const {
    return(Tensor2<double>(126., 126.));
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
Tensor2<double> InN::Me(double T, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G') {
        tMe.c00 = 0.065;
        tMe.c11 = 0.068;
    }
    return (tMe);
}

MI_PROPERTY(InN, Mhh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
Tensor2<double> InN::Mhh(double T, char point) const {
    Tensor2<double> tMhh(0.,0.);
    if (point == 'G') {
        tMhh.c00 = 1.8116;
        tMhh.c11 = 1.7007;
    }
    return (tMhh);
}

MI_PROPERTY(InN, Mlh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
Tensor2<double> InN::Mlh(double T, char point) const {
    Tensor2<double> tMlh(0.,0.);
    if (point == 'G') {
        tMlh.c00 = 1.8116;
        tMlh.c11 = 0.0348;
    }
    return (tMlh);
}

MaterialsDB::Register<InN> materialDB_register_InN;

}       // namespace plask
