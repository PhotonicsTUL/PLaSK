#include "InN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

#include <cmath>

namespace plask { namespace materials {

std::string InN::name() const { return NAME; }

MI_PROPERTY(InN, thermk,
            MISource("H. Tong et al., Proc. SPIE 7602 (2010) 76020U")
            )
Tensor2<double> InN::thermk(double T, double) const {
    return(Tensor2<double>(126. * pow((T/300.),-1.43)));
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
    if (point == 'G' || point == '*') tEg = phys::Varshni(0.69,0.414e-3,154.,T);
    return (tEg);
}

/*double InN::VB(double T, double e, char point, char hole) const {
    return 0.848;
}*/

double InN::Dso(double T, double e) const {
    return 0.005;
}


MI_PROPERTY(InN, Me,
            MISource("Adachi WILEY 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InN::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G' || point == '*') {
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

MI_PROPERTY(InN, CB,
            MISource("-")
           )
double InN::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point, 'H') + Eg(T,0.,point) );
    /*if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;stary kod z GaAs*/
}

MI_PROPERTY(InN, VB,
            MISource("-"),
            MIComment("no temperature dependence")
           )
double InN::VB(double T, double e, char point, char hole) const {
    double tVB(1.85);
    if (e) {
        /*double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L"); stary kod z GaAs*/
    }
    return tVB;
}

bool InN::isEqual(const Material &other) const {
    return true;
}

MaterialsDB::Register<InN> materialDB_register_InN;

}}       // namespace plask::materials
