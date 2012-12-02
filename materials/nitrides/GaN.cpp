#include "GaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaN::name() const { return NAME; }

MI_PROPERTY(GaN, cond,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("G. Koblmuller et al., Appl. Phys. Lett. 91 (2007) 221905"),
            MIArgumentRange(MaterialInfo::T, 270, 400)
            )
Tensor2<double> GaN::cond(double T) const {
    double tCond = 255*pow((T/300.),-0.18);
    return (Tensor2<double>(tCond,tCond));
}

MI_PROPERTY(GaN, thermk,
            MISource("C. Mion et al., App. Phys. Lett. 89 (2006) 092123"),
            MIArgumentRange(MaterialInfo::T, 300, 450)
            )
Tensor2<double> GaN::thermk(double T, double t) const {
    double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12), //TODO change t to microns
           tCondT = 230*fun_t*pow((T/300.),-1.43);
    return(Tensor2<double>(tCondT,tCondT));
 }

MI_PROPERTY(GaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double GaN::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - Eg(T,'G');
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

MI_PROPERTY(GaN, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double GaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 3.1896;
    else if (x == 'c') tLattC = 5.1855;
    return (tLattC);
}

MI_PROPERTY(GaN, Eg,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double GaN::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 3.42;
    return (tEg);
}

MI_PROPERTY(GaN, Me,
            MISource("King et al., Phys. Rev. B 75 (2007) 115312"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaN::Me(double T, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G') {
        tMe.c00 = 0.186;
        tMe.c11 = 0.209;
    }
    return (tMe);
}

MI_PROPERTY(GaN, Mhh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
Tensor2<double> GaN::Mhh(double T, char point) const {
    Tensor2<double> tMhh(0.,0.);
    if (point == 'G') {
        tMhh.c00 = 1.886;
        tMhh.c11 = 1.887;
    }
    return (tMhh);
}

MI_PROPERTY(GaN, Mlh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
Tensor2<double> GaN::Mlh(double T, char point) const {
    Tensor2<double> tMlh(0.,0.);
    if (point == 'G') {
        tMlh.c00 = 1.887;
        tMlh.c11 = 0.1086;
    }
    return (tMlh);
}

static MaterialsDB::Register<GaN> materialDB_register_GaN;

}       // namespace plask
