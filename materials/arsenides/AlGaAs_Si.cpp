#include "AlGaAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaAs_Si::name() const { return NAME; }

std::string AlGaAs_Si::str() const { return StringBuilder("Al", Al)("Ga")("As").dopant("Si", ND); }

MI_PARENT(AlGaAs_Si, AlGaAs)

AlGaAs_Si::AlGaAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaAs(Comp), mGaAs_Si(Type,Val), mAlAs_Si(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION) {//TO CHECK!!!
        Nf_RT = Val;
        if (Al < 0.35) ND = mGaAs_Si.Dop()*(1 - 7.8*Al*Al);
        else ND = mGaAs_Si.Dop()*(1.14*Al - 0.36);
    }
    else {
        ND = Val;
        if (Al < 0.35) Nf_RT = mGaAs_Si.Nf(300)*(1 - 7.8*Al*Al);
        else Nf_RT = mGaAs_Si.Nf(300)*(1.14*Al - 0.36);
    }
}

MI_PROPERTY(AlGaAs_Si, mob,
            MISource("based on 3 papers 1982-1990 about Si-doped AlGaAs"),
            MISource("based on Si-doped GaAs")
            )
std::pair<double,double> AlGaAs_Si::mob(double T) const {
    double lMob(0.), vMob(0.);
    if (Al < 0.5) {
        lMob = mGaAs_Si.mob(T).first*exp(-16*Al*Al);
        vMob = mGaAs_Si.mob(T).second*exp(-16*Al*Al);
    }
    else {
        lMob = mGaAs_Si.mob(T).first*(0.054*Al - 0.009);
        lMob = mGaAs_Si.mob(T).second*(0.054*Al - 0.009);
    }
    return (std::make_pair(lMob, vMob));
}

MI_PROPERTY(AlGaAs_Si, Nf,
            MISource("based on 2 papers 1982, 1984 about Si-doped AlGaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double AlGaAs_Si::Dop() const {
    return ND;
}

std::pair<double,double> AlGaAs_Si::cond(double T) const {
    return (std::make_pair(1.602E-17*Nf(T)*mob(T).first, 1.602E-17*Nf(T)*mob(T).second));
}

static MaterialsDB::Register<AlGaAs_Si> materialDB_register_AlGaAs_Si;

}       // namespace plask
