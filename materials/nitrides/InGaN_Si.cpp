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
double InGaN_Si::mob(double T) const {
    return ( 1/(In/mInN_Si.mob(T) + Ga/mGaN_Si.mob(T) + In*Ga*(-4.615E-21*Nf(T)+0.549)) );
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

double InGaN_Si::cond(double T) const {
    return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(InGaN_Si, condT,
            MISeeClass<InGaN>(MaterialInfo::condT),
            MIComment("Si doping dependence for GaN")
            )
double InGaN_Si::condT(double T, double t) const {
    return( 1/(In/mInN_Si.condT(T) + Ga/mGaN_Si.condT(T,t) + In*Ga*0.215*exp(7.913*In)) );
 }

MI_PROPERTY(InGaN_Si, absp,
            MISeeClass<InGaN>(MaterialInfo::absp)
            )
double InGaN_Si::absp(double wl, double T) const {
    double Eg = 0.77*In + 3.42*Ga - 1.43*In*Ga;
    double a = 1239.84190820754/wl - Eg;
    return ( (19000+4000*ND)*exp(a/(0.019+0.001*ND)) + (330+200*ND)*exp(a/(0.07+0.016*ND)) );
}

static MaterialsDB::Register<InGaN_Si> materialDB_register_InGaN_Si;

}       // namespace plask
