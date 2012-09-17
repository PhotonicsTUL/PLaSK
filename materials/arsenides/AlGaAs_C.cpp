#include "AlGaAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaAs_C::name() const { return NAME; }

std::string AlGaAs_C::str() const { return StringBuilder("Al", Al)("Ga")("As").dopant("C", NA); }

MI_PARENT(AlGaAs_C, AlGaAs)

AlGaAs_C::AlGaAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaAs(Comp), mGaAs_C(Type,Val), mAlAs_C(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.92;
    }
    else {
        NA = Val;
        Nf_RT = 0.92*Val;
    }
}

MI_PROPERTY(AlGaAs_C, mob,
            MISource("based on 4 papers 1991-2000 about C-doped AlGaAs"),
            MISource("based on C-doped GaAs")
            )
std::pair<double,double> AlGaAs_C::mob(double T) const {
    double lMob(0.), vMob(0.);
    lMob = mGaAs_C.mob(T).first*0; //TODO!!!
    vMob = mGaAs_C.mob(T).second*0; //TODO!!!
    return (std::make_pair(lMob, vMob));
}

MI_PROPERTY(AlGaAs_C, Nf,
            MISource("based on 3 papers 1991-2004 about C-doped AlGaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs_C::Nf(double T) const {
    return ( Nf_RT );
}

double AlGaAs_C::Dop() const {
    return NA;
}

std::pair<double,double> AlGaAs_C::cond(double T) const {
    return (std::make_pair(1.602E-17*Nf(T)*mob(T).first, 1.602E-17*Nf(T)*mob(T).second));
}

static MaterialsDB::Register<AlGaAs_C> materialDB_register_AlGaAs_C;

}       // namespace plask
