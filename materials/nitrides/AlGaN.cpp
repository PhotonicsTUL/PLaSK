#include "AlGaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaN::name() const { return NAME; }

std::string AlGaN::str() const { return StringBuilder("Al", Al)("Ga")("N"); }

AlGaN::AlGaN(const Material::Composition& Comp) {

    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(AlGaN, thermk,
            MISource("B. C. Daly et al., Journal of Applied Physics 92 (2002) 3820"),
            MIComment("based on data for Al = 0.2, 0.45")
            )
Tensor2<double> AlGaN::thermk(double T, double t) const {
    double lCondT = 1/(Al/mAlN.thermk(T,t).c00 + Ga/mGaN.thermk(T,t).c00 + Al*Ga*0.4),
           vCondT = 1/(Al/mAlN.thermk(T,t).c11 + Ga/mGaN.thermk(T,t).c11 + Al*Ga*0.4);
    return Tensor2<double>(lCondT,vCondT);
 }

MI_PROPERTY(AlGaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double AlGaN::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - Eg(T,'G');
    return 19000*exp(a/0.019) + 330*exp(a/0.07);
}

MI_PROPERTY(AlGaN, nr,
            MISource("www.rpi.edu Educational Resources (E.F. Schubert 2004)"),
            MIArgumentRange(MaterialInfo::wl, 355, 1240),
            MIComment("based on data for Al = 0, 0.3, 1.0"),
            MIComment("no temperature dependence")
            )
double AlGaN::nr(double wl, double T) const {
    double E = 1239.84190820754/wl,
           a =  0.073-Al*0.042-Al*Al*0.014,
           b = -0.179-Al*0.032+Al*Al*0.174,
           c =  2.378-Al*0.354-Al*Al*0.017;
    return ( a*E*E + b*E + c );
}

MI_PROPERTY(AlGaN, Eg,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double AlGaN::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 6.28*Al + 3.42*Ga - 0.7*Al*Ga;
    return (tEg);
}

MI_PROPERTY(AlGaN, lattC,
            MISource("linear interpolation: GaN, AlN")
            )
double AlGaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = mAlN.lattC(T,'a')*Al + mGaN.lattC(T,'a')*Ga;
    else if (x == 'c') tLattC = mAlN.lattC(T,'c')*Al + mGaN.lattC(T,'c')*Ga;
    return (tLattC);
}

bool AlGaN::isEqual(const Material &other) const {
    const AlGaN& o = static_cast<const AlGaN&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaN> materialDB_register_AlGaN;

}       // namespace plask
