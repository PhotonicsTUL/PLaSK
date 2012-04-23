#include "AlN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlN_Mg::name() const { return NAME; }

std::string AlN_Mg::str() const { return StringBuilder("AlN").dopant("Mg", NA); }

MI_PARENT(AlN_Mg, AlN)

AlN_Mg::AlN_Mg(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = 6.4e9;
    }
    else {
        Nf_RT = 6.4e9;
        NA = Val;
    }
    //mobRT(Nf_RT),
    mob_RT = 1.7;
}

MI_PROPERTY(AlN_Mg, mob,
            MISource("Y. Taniyasu et al., Nature Letters 44 (2006) 325"),
            MIArgumentRange(MaterialInfo::T, 300, 400)
			)
double AlN_Mg::mob(double T) const {
    return ( 5.377e3*pow(mob_RT,-1.506) );
}

MI_PROPERTY(AlN_Mg, Nf,
            MISource("Taniyasu Y, Nature Letters 44 (2006) 325"),
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
double AlN_Mg::Nf(double T) const {
    return ( 6.689e-55*pow(Nf_RT,2.187e1) );
}

double AlN_Mg::Dop() const {
    return NA;
}

MI_PROPERTY(AlN_Mg, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
double AlN_Mg::cond(double T) const {
	return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(AlN_Mg, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("no temperature dependence")
            )
double AlN_Mg::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - 6.28,
           b = NA/1e18;
    return ( (19000+200*b)*exp(a/(0.019+0.0001*b)) + (330+30*b)*exp(a/(0.07+0.0008*b)) );
}

static MaterialsDB::Register<AlN_Mg> materialDB_register_AlN_Mg;

}       // namespace plask
