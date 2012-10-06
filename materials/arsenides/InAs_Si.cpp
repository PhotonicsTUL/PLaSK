#include "InAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InAs_Si::name() const { return NAME; }

std::string InAs_Si::str() const { return StringBuilder("InAs").dopant("Si", ND); }

InAs_Si::InAs_Si(DopingAmountType Type, double Val) {
    Nf_RT = 0.; //TODO
    ND = 0.; //TODO
    mob_RT = 0.; //TODO
}

MI_PROPERTY(InAs_Si, mob,
            MIComment("TODO")
            )
std::pair<double,double> InAs_Si::mob(double T) const {
    return ( std::make_pair(mob_RT,mob_RT) );
}

MI_PROPERTY(InAs_Si, Nf,
            MIComment("TODO")
            )
double InAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double InAs_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InAs_Si, cond,
			MIComment("no temperature dependence")
            )
std::pair<double,double> InAs_Si::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( std::make_pair(tCond, tCond) );
}

static MaterialsDB::Register<InAs_Si> materialDB_register_InAs_Si;

} // namespace plask
