#include "AlAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlAs_Si::name() const { return NAME; }

std::string AlAs_Si::str() const { return StringBuilder("AlAs").dopant("Si", ND); }

AlAs_Si::AlAs_Si(DopingAmountType Type, double Val) {
    //TODO
    Nf_RT = 0;
    ND = 0;
    mob_RT = 0;
}

MI_PROPERTY(AlAs_Si, mob,
            MIComment("TODO")
            )
std::pair<double,double> AlAs_Si::mob(double T) const {
    return (std::make_pair(mob_RT,mob_RT));
}

MI_PROPERTY(AlAs_Si, Nf,
            MIComment("TODO")
            )
double AlAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double AlAs_Si::Dop() const {
    return ND;
}

MI_PROPERTY(AlAs_Si, cond,
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlAs_Si::cond(double T) const {
    double tCond = 1.602e-17*Nf_RT*mob_RT;
    return (std::make_pair(tCond, tCond));
}


static MaterialsDB::Register<AlAs_Si> materialDB_register_AlAs_Si;

}       // namespace plask
