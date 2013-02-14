#include "InGaN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

MI_PARENT(InGaN_Si, InGaN)

std::string InGaN_Si::name() const { return NAME; }

std::string InGaN_Si::str() const { return StringBuilder("In", In)("Ga")("N").dopant("Si", ND); }

InGaN_Si::InGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): InGaN(Comp), mGaN_Si(Type,Val), mInN_Si(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        ND = mInN_Si.Dop()*In + mGaN_Si.Dop()*Ga;
    else
        ND = Val;
}

MI_PROPERTY(InGaN_Si, mob,
            MISource("based on 3 papers 2007-2009 about Si-doped InGaN/GaN/c-sapphire"),
            MISource("based on Si-doped GaN and InN")
            )
Tensor2<double> InGaN_Si::mob(double T) const {
    double lMob = 1/(In/mInN_Si.mob(T).c00 + Ga/mGaN_Si.mob(T).c00 + In*Ga*(-4.615E-21*Nf(T)+0.549)),
           vMob = 1/(In/mInN_Si.mob(T).c11 + Ga/mGaN_Si.mob(T).c11 + In*Ga*(-4.615E-21*Nf(T)+0.549));
    return (Tensor2<double>(lMob, vMob));
}

MI_PROPERTY(InGaN_Si, Nf,
            MISource("linear interpolation: Si-doped GaN, InN")
            )
double InGaN_Si::Nf(double T) const {
    return ( mInN_Si.Nf(T)*In + mGaN_Si.Nf(T)*Ga );
}

double InGaN_Si::Dop() const {
    return ND;
}

Tensor2<double> InGaN_Si::cond(double T) const {
    return (Tensor2<double>(1.602E-17*Nf(T)*mob(T).c00, 1.602E-17*Nf(T)*mob(T).c11));
}

MI_PROPERTY(InGaN_Si, thermk,
            MISeeClass<InGaN>(MaterialInfo::thermk),
            MIComment("Si doping dependence for GaN")
            )
Tensor2<double> InGaN_Si::thermk(double T, double t) const {
    double lCondT = 1/(In/mInN_Si.thermk(T).c00 + Ga/mGaN_Si.thermk(T,t).c00 + In*Ga*0.215*exp(7.913*In)),
           vCondT = 1/(In/mInN_Si.thermk(T).c11 + Ga/mGaN_Si.thermk(T,t).c11 + In*Ga*0.215*exp(7.913*In));
    return(Tensor2<double>(lCondT, vCondT));
 }

MI_PROPERTY(InGaN_Si, absp,
            MISeeClass<InGaN>(MaterialInfo::absp)
            )
double InGaN_Si::absp(double wl, double T) const {
    double Eg = 0.77*In + 3.42*Ga - 1.43*In*Ga;
    double a = 1239.84190820754/wl - Eg,
           b = ND/1e18;
    return ( (19000+4000*b)*exp(a/(0.019+0.001*b)) + (330+200*b)*exp(a/(0.07+0.016*b)) );
}

bool InGaN_Si::isEqual(const Material &other) const {
    const InGaN_Si& o = static_cast<const InGaN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && InGaN::isEqual(other);
}


static MaterialsDB::Register<InGaN_Si> materialDB_register_InGaN_Si;

}       // namespace plask
