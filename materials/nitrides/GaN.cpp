#include "GaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaN::name() const { return NAME; }

MI_PROPERTY(GaN, cond,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("G. Koblmï¿½ller et al., Appl. Phys. Lett. 91 (2007) 221905"),
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
double GaN::Me(double T, char point) const {
    double tMe(0.);
    if (point == 'G') tMe = 0.2;
    return (tMe);
}

MI_PROPERTY(GaN, Me_v,
            MISource("Yan et al., Sem. Sci. Technol. 26 (2011) 014037"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double GaN::Me_v(double T, char point) const {
    double tMe_v(0.);
    if (point == 'G') tMe_v = 0.209;
    return (tMe_v);
}

MI_PROPERTY(GaN, Me_l,
            MISeeClass<GaN>(MaterialInfo::Me_v)
            )
double GaN::Me_l(double T, char point) const {
    double tMe_l(0.);
    if (point == 'G') tMe_l = 0.186;
    return (tMe_l);
}

MI_PROPERTY(GaN, Mhh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
double GaN::Mhh(double T, char point) const {
    double tMhh(0.);
    if (point == 'G') tMhh = 1.887;
    return (tMhh);
}

MI_PROPERTY(GaN, Mhh_v,
            MISeeClass<GaN>(MaterialInfo::Me_v)
            )
double GaN::Mhh_v(double T, char point) const {
    double tMhh_v(0.);
    if (point == 'G') tMhh_v = 1.887;
    return (tMhh_v);
}

MI_PROPERTY(GaN, Mhh_l,
            MISeeClass<GaN>(MaterialInfo::Me_v)
            )
double GaN::Mhh_l(double T, char point) const {
    double tMhh_l(0.);
    if (point == 'G') tMhh_l = 1.876;
    return (tMhh_l);
}

MI_PROPERTY(GaN, Mlh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
double GaN::Mlh(double T, char point) const {
    double tMlh(0.);
    if (point == 'G') tMlh = 1.887;
    return (tMlh);
}

MI_PROPERTY(GaN, Mlh_v,
            MISeeClass<GaN>(MaterialInfo::Me_v)
            )
double GaN::Mlh_v(double T, char point) const {
    double tMlh_v(0.);
    if (point == 'G') tMlh_v = 0.1086;
    return (tMlh_v);
}

MI_PROPERTY(GaN, Mlh_l,
            MISeeClass<GaN>(MaterialInfo::Me_v)
            )
double GaN::Mlh_l(double T, char point) const {
    double Mlh_l(0.);
    if (point == 'G') Mlh_l = 1.876;
    return (Mlh_l);
}

static MaterialsDB::Register<GaN> materialDB_register_GaN;

}       // namespace plask
