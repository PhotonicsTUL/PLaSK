#include "GaAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaAs_C::name() const { return NAME; }

std::string GaAs_C::str() const { return StringBuilder("GaAs").dopant("C", NA); }

GaAs_C::GaAs_C(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.92;
    }
    else {
        Nf_RT = 0.92*Val;
        NA = Val;
    }
    mob_RT = 530/(1+pow((Nf_RT/1e17),0.3));
}

MI_PROPERTY(GaAs_C, mob,
            MISource("fit to p-GaAs:C (based on 23 papers 1988 - 2006)"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs_C::mob(double T) const {
    return (std::make_pair(mob_RT,mob_RT));
}


MI_PROPERTY(GaAs_C, cond,
			MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs_C::cond(double T) const {
    double tCond = 1.602e-17*Nf_RT*mob_RT;
    return (std::make_pair(tCond, tCond));
}

MI_PROPERTY(GaAs_C, absp,
            MISource(""),
            MIComment("TODO")
            )
double GaAs_C::absp(double wl, double T) const {
    return ( 0 );
}


static MaterialsDB::Register<GaAs_C> materialDB_register_GaAs_C;

}       // namespace plask
