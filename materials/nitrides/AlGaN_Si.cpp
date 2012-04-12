#include "AlGaN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaN_Si::name() const { return NAME; }

MI_PARENT(AlGaN_Si, AlGaN)

AlGaN_Si::AlGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaN(Comp)
{
    mGaN_Si = new GaN_Si(Type, Val);
    mAlN_Si = new AlN_Si(Type, Val);

    if (Type == CARRIER_CONCENTRATION)
        ND = mAlN_Si->Dop()*Al + mGaN_Si->Dop()*Ga;
    else
        ND = Val;
}

AlGaN_Si::~AlGaN_Si()
{
    delete mGaN_Si;
    delete mAlN_Si;
}

MI_PROPERTY(AlGaN_Si, mob,
            MISource("based on 11 papers 1997-2008 about Si-doped AlGaN"),
            MISource("based on Si-doped GaN and AlN")
            )
double AlGaN_Si::mob(double T) const {
    return ( Al*mAlN_Si->mob(T) + pow(Ga,1.415+19.63*exp(-5.456*Al))*mGaN_Si->mob(T) );
}

MI_PROPERTY(AlGaN_Si, Nf,
            MISource("linear interpolation: Si-doped GaN, AlN")
            )
double AlGaN_Si::Nf(double T) const {
    return ( mAlN_Si->Nf(T)*Al + mGaN_Si->Nf(T)*Ga );
}

double AlGaN_Si::Dop() const {
    return ND;
}

double AlGaN_Si::cond(double T) const {
    return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(AlGaN_Si, condT,
            MISeeClass<AlGaN>(MaterialInfo::condT),
            MIComment("Si doping dependence for GaN")
            )
double AlGaN_Si::condT(double T, double t) const {
    return( 1/(Al/mAlN_Si->condT(T,t) + Ga/mGaN_Si->condT(T,t) + Al*Ga*0.4) );
 }

MI_PROPERTY(AlGaN_Si, absp,
            MISeeClass<AlGaN>(MaterialInfo::absp)
            )
double AlGaN_Si::absp(double wl, double T) const {
    double Eg = 6.28*Al + 3.42*(1-Al) - 0.7*Al*(1-Al);
    double a = 1239.84190820754/wl - Eg;
    return ( (19000+4000*ND)*exp(a/(0.019+0.001*ND)) + (330+200*ND)*exp(a/(0.07+0.016*ND)) );
}

static MaterialsDB::Register<AlGaN_Si> materialDB_register_AlGaN_Si;

}       // namespace plask
