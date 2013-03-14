#include "InGaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InGaN::name() const { return NAME; }

std::string InGaN::str() const { return StringBuilder("In", In)("Ga")("N"); }

InGaN::InGaN(const Material::Composition& Comp)
{
    In = Comp.find("In")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(InGaN, thermk,
            MISource("B. N. Pantha et al., Applied Physics Letters 92 (2008) 042112"),
            MIComment("based on data for In: 16% - 36%")
            )
Tensor2<double> InGaN::thermk(double T, double t) const {
    double lCondT = 1/(In/mInN.thermk(T).c00 + Ga/mGaN.thermk(T,t).c00 + In*Ga*0.215*exp(7.913*In)),
           vCondT = 1/(In/mInN.thermk(T).c11 + Ga/mGaN.thermk(T,t).c11 + In*Ga*0.215*exp(7.913*In));
    return(Tensor2<double>(lCondT,vCondT));
 }

MI_PROPERTY(InGaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double InGaN::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - Eg(T, 0, 'G');
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(InGaN, nr,
            MISource("N. A. Sanford et al., Phys. Status Solidi C 2 (2005) 2783"),
            MIArgumentRange(MaterialInfo::wl, 460, 530),
            MIComment("based on data for In: 0% - 6%")
            )
double InGaN::nr(double wl, double T) const {
    double a = -0.00006862*pow(wl,3) + 0.0978731*pow(wl,2) - 46.3535*wl + 7287.33,
           b =  0.00000429*pow(wl,3) - 0.0059179*pow(wl,2) + 2.69562*wl - 404.952,
           c = -0.00000006*pow(wl,3) + 0.0000872*pow(wl,2) - 0.04383*wl + 9.87511;
    return ( a*In*In+b*In+c );
}

MI_PROPERTY(InGaN, Eg,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
double InGaN::Eg(double T, double eps, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 0.77*In + 3.42*Ga - 1.43*In*Ga;
    return (tEg);
}

MI_PROPERTY(InGaN, lattC,
            MISource("linear interpolation: GaN, InN")
            )
double InGaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = mInN.lattC(T,'a')*In + mGaN.lattC(T,'a')*Ga;
    else if (x == 'c') tLattC = mInN.lattC(T,'c')*In + mGaN.lattC(T,'c')*Ga;
    return (tLattC);
}

bool InGaN::isEqual(const Material &other) const {
    const InGaN& o = static_cast<const InGaN&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<InGaN> materialDB_register_InGaN;

}       // namespace plask

