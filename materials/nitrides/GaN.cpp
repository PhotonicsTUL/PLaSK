#include "GaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaN::name() const { return NAME; }

MI_PROPERTY(GaN, cond,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("G. Koblm�ller et al., Appl. Phys. Lett. 91 (2007) 221905"),
            MIArgumentRange(MaterialInfo::T, 270, 400)
            )
double GaN::cond(double T) const {
    return ( 255*pow((T/300.),-0.18) );
}

MI_PROPERTY(GaN, condT,
            MISource("C. Mion et al., App. Phys. Lett. 89 (2006) 092123"),
            MIArgumentRange(MaterialInfo::T, 300, 450)
            )
double GaN::condT(double T, double t) const {
	double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12);
    return( 230*fun_t*pow((T/300.),-1.43) );
 }

MI_PROPERTY(GaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("no temperature dependence")
            )
double GaN::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - 3.42;
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(GaN, nr,
            MISource("www.rpi.edu Educational Resources (E.F. Schubert 2004)"),
            MIArgumentRange(MaterialInfo::wl, 355, 1240),
            MIComment("no temperature dependence")
            )
double GaN::nr(double wl, double T) const {
    double a = 1239.84190820754/wl;
    return ( 0.0259462*pow(a,3) - 0.101517*pow(a,2) + 0.181516*a + 2.1567 );
}

static MaterialsDB::Register<GaN> materialDB_register_GaN;

}       // namespace plask
