#include "InGaN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

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
Tensor2<double> InGaN_Mg::mob(double T) const {
    double lMob = 1/(In/mInN_Mg.mob(T).c00 + Ga/mGaN_Mg.mob(T).c00 + In*Ga*(7.256E-19*Nf(T)+0.377)),
           vMob = 1/(In/mInN_Mg.mob(T).c11 + Ga/mGaN_Mg.mob(T).c11 + In*Ga*(7.256E-19*Nf(T)+0.377));
    return (Tensor2<double>(lMob, vMob));
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

Tensor2<double> InGaN_Mg::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

MI_PROPERTY(InGaN_Mg, absp,
            MISeeClass<InGaN>(MaterialInfo::absp)
            )
double InGaN_Mg::absp(double wl, double T) const {
    double E = phys::h_eVc1e9/wl;
    return ( (19000.+200.*Dop()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.0001*Dop()/1e18))+(330.+30.*Dop()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.0008*Dop()/1e18)) );
}

bool InGaN_Mg::isEqual(const Material &other) const {
    const InGaN_Mg& o = static_cast<const InGaN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && InGaN::isEqual(other);
}

static MaterialsDB::Register<InGaN_Mg> materialDB_register_InGaN_Mg;

}}       // namespace plask::materials
