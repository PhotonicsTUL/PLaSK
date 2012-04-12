#include "AlN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlN_Si::name() const { return NAME; }

MI_PARENT(AlN_Si, AlN)

AlN_Si::AlN_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        ND = 6.197E-19*pow(Val,1.805);
    }
    else {
        Nf_RT = 1.223e10*pow(Val,5.540e-1);;
        ND = Val;
    }
    //mobRT(Nf_RT),
    mob_RT = 29.410*exp(-1.838E-17*Nf_RT);
}

MI_PROPERTY(AlN_Si, mob,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MIComment("based on 4 papers (2004-2008): Si-doped AlN")
			)
double AlN_Si::mob(double T) const {
    return ( mob_RT * (1.486 -T*0.00162) );
}

MI_PROPERTY(AlN_Si, Nf,
            MISource("Y. Taniyasu, Nature Letters 44 (2006) 325"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("based on 2 papers (2004-2008): Si-doped AlN")
            )
double AlN_Si::Nf(double T) const {
	return ( Nf_RT * 3.502E-27*pow(T,10.680) );
}

double AlN_Si::Dop() const {
    return ND;
}

MI_PROPERTY(AlN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
double AlN_Si::cond(double T) const {
	return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(AlN_Si, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("no temperature dependence")
            )
double AlN_Si::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - 6.28;
    return ( (19000+400*ND)*exp(a/(0.019+0.001*ND)) + (330+200*ND)*exp(a/(0.07+0.016*ND)) );
}

static MaterialsDB::Register<AlN_Si> materialDB_register_AlN_Si;

}       // namespace plask
