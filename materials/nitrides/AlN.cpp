#include "AlN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlN::name() const { return NAME; }

MI_PROPERTY(AlN, thermk,
            MISource("G. A. Slack, J. Phys. Chem. Sol. 48 (1987) 641"),
            MISource("Bondokov R T, J. Crystal Growth 310 (2008) 4020"),
            MISource("M. Kuc, R.P. Sarzala and W. Nakwaski, Materials Science and Engineering B, 178 (2013) 1395-1402"))
Tensor2<double> AlN::thermk(double T, double /*t*/) const {
    double tCondT = 270.*pow((T/300.),-1.25);
    return(Tensor2<double>(tCondT,tCondT));
 }

MI_PROPERTY(AlN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double AlN::absp(double lam, double T) const {
    double a = phys::h_eVc1e9/lam - Eg(T,0.,'G');
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(AlN, nr,
            MISource("www.rpi.edu Educational Resources (E.F. Schubert 2004)"),
            MIArgumentRange(MaterialInfo::lam, 225, 1240),
            MIComment("no temperature dependence")
            )
double AlN::nr(double lam, double /*T*/, double /*n*/) const {
    double a = phys::h_eVc1e9/lam;
    return ( 0.0034417*pow(a,3) - 0.0172622*pow(a,2) + 0.0594128*a + 1.92953 );
}

MI_PROPERTY(AlN, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double AlN::lattC(double /*T*/, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 3.112;
    else if (x == 'c') tLattC = 4.982;
    return (tLattC);
}

MI_PROPERTY(AlN, Eg,
            MISource("Vurgaftman et al. in Piprek 2007 Nitride Semicondcuctor Devices")
            )
double AlN::Eg(double T, double /*e*/, char point) const {
    double tEg(0.);
    if (point == 'G' || point == '*') tEg = phys::Varshni(6.10,2.63e-3,2082.,T);
    return (tEg);
}

double AlN::VB(double /*T*/, double /*e*/, char /*point*/, char /*hole*/) const {
    return -0.769;
}

double AlN::Dso(double /*T*/, double /*e*/) const {
    return 0.036;
}

MI_PROPERTY(AlN, Me,
            MISource("Adachi WILEY 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlN::Me(double /*T*/, double /*e*/, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G' || point == '*') {
        tMe.c00 = 0.30;
        tMe.c11 = 0.29;
    }
    return (tMe);
}

MI_PROPERTY(AlN, Mhh,
            MISeeClass<AlN>(MaterialInfo::Me)
            )
Tensor2<double> AlN::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(2.56, 4.17);
}

MI_PROPERTY(AlN, Mlh,
            MISeeClass<AlN>(MaterialInfo::Me)
            )
Tensor2<double> AlN::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(2.56, 0.27);
}

Material::ConductivityType AlN::condtype() const { return Material::CONDUCTIVITY_I; }

bool AlN::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<AlN> materialDB_register_AlN;

}}       // namespace plask::materials
