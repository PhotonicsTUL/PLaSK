#include "InGaN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

MI_PARENT(InGaN_Mg, InGaN)

std::string InGaN_Mg::name() const { return NAME; }

std::string InGaN_Mg::str() const { return StringBuilder("In", In)("Ga")("N").dopant("Mg", NA); }

InGaN_Mg::InGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val): InGaN(Comp), mGaN_Mg(Type,Val), mInN_Mg(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        NA = mInN_Mg.Dop()*In + mGaN_Mg.Dop()*Ga;
    else
        NA = Val;
}

MI_PROPERTY(InGaN_Mg, mob,
            MISource("B. N. Pantha et al., Applied Physics Letters 95 (2009) 261904"),
            MISource("K. Aryal et al., Applied Physics Letters 96 (2010) 052110")
            )
double InGaN_Mg::mob(double T) const {
    return ( 1/(In/mInN_Mg.mob(T) + Ga/mGaN_Mg.mob(T) + In*Ga*(7.256E-19*Nf(T)+0.377)) );
}

MI_PROPERTY(InGaN_Mg, Nf,
            MISource("linear interpolation: Mg-doped GaN, InN")
            )
double InGaN_Mg::Nf(double T) const {
    return ( mInN_Mg.Nf(T)*In + mGaN_Mg.Nf(T)*Ga );
}

double InGaN_Mg::Dop() const {
    return NA;
}

double InGaN_Mg::cond(double T) const {
    return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(InGaN_Mg, absp,
            MISeeClass<InGaN>(MaterialInfo::absp)
            )
double InGaN_Mg::absp(double wl, double T) const {
    double Eg = 0.77*In + 3.42*Ga - 1.43*In*Ga;
    double a = 1239.84190820754/wl - Eg;
    return ( (19000+200*NA)*exp(a/(0.019+0.0001*NA)) + (330+30*NA)*exp(a/(0.07+0.0008*NA)) );
}

static MaterialsDB::Register<InGaN_Mg> materialDB_register_InGaN_Mg;

}       // namespace plask
