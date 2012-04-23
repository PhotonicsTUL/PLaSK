#include "AlGaN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaN_Mg::name() const { return NAME; }

std::string AlGaN_Mg::str() const { return StringBuilder("Al", Al)("Ga")("N").dopant("Mg", NA); }

MI_PARENT(AlGaN_Mg, AlGaN)

AlGaN_Mg::AlGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaN(Comp), mGaN_Mg(Type,Val), mAlN_Mg(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        NA = mAlN_Mg.Dop()*Al + mGaN_Mg.Dop()*Ga;
    else
        NA = Val;
}

MI_PROPERTY(AlGaN_Mg, mob,
            MISource("based on 7 papers 1994-2010 about Mg-doped AlGaN"),
            MISource("based on Mg-doped GaN and AlN")
            )
double AlGaN_Mg::mob(double T) const {
    return ( pow(Ga,28.856-16.793*(1-exp(-Al/0.056))-9.259*(1-exp(-Al/0.199)))*mGaN_Mg.mob(T) );
}

MI_PROPERTY(AlGaN_Mg, Nf,
            MISource("linear interpolation: Mg-doped GaN, AlN")
            )
double AlGaN_Mg::Nf(double T) const {
    return ( mAlN_Mg.Nf(T)*Al + mGaN_Mg.Nf(T)*Ga );
}

double AlGaN_Mg::Dop() const {
    return NA;
}

double AlGaN_Mg::cond(double T) const {
    return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(AlGaN_Mg, absp,
            MISeeClass<AlGaN>(MaterialInfo::absp)
            )
double AlGaN_Mg::absp(double wl, double T) const {
    double Eg = 6.28*Al + 3.42*(1-Al) - 0.7*Al*(1-Al);
    double a = 1239.84190820754/wl - Eg,
           b = NA/1e18;
    return ( (19000+200*b)*exp(a/(0.019+0.0001*b)) + (330+30*b)*exp(a/(0.07+0.0008*b)) );
}

static MaterialsDB::Register<AlGaN_Mg> materialDB_register_AlGaN_Mg;

}       // namespace plask
