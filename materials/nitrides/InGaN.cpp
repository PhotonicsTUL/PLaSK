#include "InGaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InGaN::name() const { return NAME; }


InGaN::InGaN(const Material::Composition& Comp)
{
    mGaN = new GaN();
    mInN = new InN();

    In = Comp.find("In")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(InGaN, condT,
            MISource("B. N. Pantha et al., Applied Physics Letters 92 (2008) 042112"),
            MIComment("based on data for In: 16% - 36%")
            )
double InGaN::condT(double T, double t) const {
    return( 1/(In/mInN->condT(T) + Ga/mGaN->condT(T,t) + In*Ga*0.215*exp(7.913*In)) );
 }

MI_PROPERTY(InGaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("no temperature dependence")
            )
double InGaN::absp(double wl, double T) const {
    double Eg = 0.77*In + 3.42*Ga - 1.43*In*Ga;
    double a = 1239.84190820754/wl - Eg;
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(InGaN, nr,
            MISource(""),
            MIArgumentRange(MaterialInfo::wl, 0, 0),
            MIComment("")
            )
double InGaN::nr(double wl, double T) const {
    return ( 0. );  //TODO!!!
}

static MaterialsDB::Register<InGaN> materialDB_register_InGaN;

}       // namespace plask
