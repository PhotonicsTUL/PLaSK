#include "AlN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlN::name() const { return NAME; }

MI_PROPERTY(AlN, thermk,
            MISource("G. A. Slack, J. Phys. Chem. Sol. 48 (1987) 641"),
            MISource("Bondokov R T, J. Crystal Growth 310 (2008) 4020"),
            MIComment("based on Si-doped GaN and AlN data to estimate thickness dependence"))
Tensor2<double> AlN::thermk(double T, double t) const {
    double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12), //TODO change t to microns
           tCondT = 285*fun_t*pow((T/300.),-1.25);
    return(Tensor2<double>(tCondT,tCondT));
 }

MI_PROPERTY(AlN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double AlN::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - Eg(T,0.,'G');
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(AlN, nr,
            MISource("www.rpi.edu Educational Resources (E.F. Schubert 2004)"),
            MIArgumentRange(MaterialInfo::wl, 225, 1240),
            MIComment("no temperature dependence")
            )
double AlN::nr(double wl, double T) const {
    double a = 1239.84190820754/wl;
    return ( 0.0034417*pow(a,3) - 0.0172622*pow(a,2) + 0.0594128*a + 1.92953 );
}

MI_PROPERTY(AlN, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double AlN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 3.112;
    else if (x == 'c') tLattC = 4.982;
    return (tLattC);
}

MI_PROPERTY(AlN, Eg,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double AlN::Eg(double T, double eps, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 6.28;
    return (tEg);
}

MI_PROPERTY(AlN, Me,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlN::Me(double T, double eps, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G') {
        tMe.c00 = 0.30;
        tMe.c11 = 0.29;
    }
    return (tMe);
}

bool AlN::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<AlN> materialDB_register_AlN;

}       // namespace plask
