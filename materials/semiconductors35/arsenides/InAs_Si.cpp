#include "InAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

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
Tensor2<double> InAs_Si::mob(double T) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
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
Tensor2<double> InAs_Si::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob;
    return ( Tensor2<double>(tCond, tCond) );
}

bool InAs_Si::isEqual(const Material &other) const {
    const InAs_Si& o = static_cast<const InAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAs::isEqual(other);
}

static MaterialsDB::Register<InAs_Si> materialDB_register_InAs_Si;

}} // namespace plask::materials
