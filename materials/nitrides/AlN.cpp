#include "AlN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlN::name() const { return NAME; }

MI_PROPERTY(AlN, condT,
            MISource("G. A. Slack, J. Phys. Chem. Sol. 48 (1987) 641"),
            MISource("Bondokov R T, J. Crystal Growth 310 (2008) 4020"),
            MIComment("based on Si-doped GaN and AlN data to estimate thickness dependence"))
double AlN::condT(double T, double t) const {
	double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12);
    return( 285*fun_t*pow((T/300.),-1.25) );
 }

MI_PROPERTY(AlN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("no temperature dependence")
            )
double AlN::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - 6.28;
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

static MaterialsDB::Register<AlN> materialDB_register_AlN;

}       // namespace plask
