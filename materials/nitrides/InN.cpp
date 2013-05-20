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
Tensor2<double> InN::thermk(double T, double) const {
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
            MISource("Vurgaftman et al. in Piprek 2007 Nitride Semicondcuctor Devices")
            )
double InN::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.69,0.414e-3,154.,T);
    return (tEg);
}

MI_PROPERTY(InN, Me,
            MISource("Adachi WILEY 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InN::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G') {
        tMe.c00 = 0.039;
        tMe.c11 = 0.047;
    }
    return (tMe);
}

MI_PROPERTY(InN, Mhh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
Tensor2<double> InN::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.,0.);
    tMhh.c00 = 1.54;
    tMhh.c11 = 1.41;
    return (tMhh);
}

MI_PROPERTY(InN, Mlh,
            MISeeClass<InN>(MaterialInfo::Me)
            )
Tensor2<double> InN::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.,0.);
    tMlh.c00 = 1.54;
    tMlh.c11 = 0.10;
    return (tMlh);
}

bool InN::isEqual(const Material &other) const {
    return true;
}

MaterialsDB::Register<InN> materialDB_register_InN;

}       // namespace plask
