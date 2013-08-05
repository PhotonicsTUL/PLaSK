#include "AlGaN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaN_Mg::name() const { return NAME; }

std::string AlGaN_Mg::str() const { return StringBuilder("Al", Al)("Ga")("N").dopant("Mg", NA); }

MI_PARENT(AlGaN_Mg, AlGaN)

AlGaN_Mg::AlGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaN(Comp), mGaN_Mg(Type,Val), mAlN_Mg(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        NA = mGaN_Mg.Dop();
        //NA = mAlN_Mg.Dop()*Al + mGaN_Mg.Dop()*Ga;
    else
        NA = Val;
}

MI_PROPERTY(AlGaN_Mg, mob,
            MISource("based on 7 papers 1994-2010 about Mg-doped AlGaN"),
            MISource("based on Mg-doped GaN and AlN")
            )
Tensor2<double> AlGaN_Mg::mob(double T) const {
    double lMob = pow(Ga,28.856-16.793*(1-exp(-Al/0.056))-9.259*(1-exp(-Al/0.199)))*mGaN_Mg.mob(T).c00,
           vMob = pow(Ga,28.856-16.793*(1-exp(-Al/0.056))-9.259*(1-exp(-Al/0.199)))*mGaN_Mg.mob(T).c11;
    return (Tensor2<double>(lMob,vMob));
}

MI_PROPERTY(AlGaN_Mg, Nf,
            MISource("linear interpolation: Mg-doped GaN, AlN")
            )
double AlGaN_Mg::Nf(double T) const {
    return mGaN_Mg.Nf(T);
    //return mAlN_Mg.Nf(T)*Al + mGaN_Mg.Nf(T)*Ga;
}

double AlGaN_Mg::Dop() const {
    return NA;
}

Tensor2<double> AlGaN_Mg::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

MI_PROPERTY(AlGaN_Mg, absp,
            MISeeClass<AlGaN>(MaterialInfo::absp)
            )
double AlGaN_Mg::absp(double wl, double T) const {
    double E = phys::h_eVc1e9/wl;
    return ( (19000.+200.*Dop()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.0001*Dop()/1e18))+(330.+30.*Dop()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.0008*Dop()/1e18)) );
}

bool AlGaN_Mg::isEqual(const Material &other) const {
    const AlGaN_Mg& o = static_cast<const AlGaN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && AlGaN::isEqual(other);
}

static MaterialsDB::Register<AlGaN_Mg> materialDB_register_AlGaN_Mg;

}}       // namespace plask::materials
