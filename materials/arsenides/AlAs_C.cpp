#include "AlAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlAs_C::name() const { return NAME; }

std::string AlAs_C::str() const { return StringBuilder("AlAs").dopant("C", NA); }

AlAs_C::AlAs_C(DopingAmountType Type, double Val) {
    //TODO
    Nf_RT = 0;
    NA = 0;
    mob_RT = 0;
}

MI_PROPERTY(AlAs_C, mob,
            MIComment("TODO")
            )
std::pair<double,double> AlAs_C::mob(double T) const {
    return (std::make_pair(mob_RT,mob_RT));
}

MI_PROPERTY(AlAs_C, Nf,
            MIComment("TODO")
            )
double AlAs_C::Nf(double T) const {
    return ( Nf_RT );
}

double AlAs_C::Dop() const {
    return NA;
}

MI_PROPERTY(AlAs_C, cond,
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlAs_C::cond(double T) const {
    double tCond = 1.602e-17*Nf_RT*mob_RT;
    return (std::make_pair(tCond, tCond));
}


static MaterialsDB::Register<AlAs_C> materialDB_register_AlAs_C;

}       // namespace plask
