#include "InAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAs_C::name() const { return NAME; }

std::string InAs_C::str() const { return StringBuilder("InAs").dopant("C", NA); }

InAs_C::InAs_C(DopingAmountType /*Type*/, double Val) {
    Nf_RT = 0.; //TODO
    NA = 0.; //TODO
    mob_RT = 0.; //TODO
}

MI_PROPERTY(InAs_C, mob,
            MIComment("TODO")
            )
Tensor2<double> InAs_C::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InAs_C, Nf,
            MIComment("TODO")
            )
double InAs_C::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double InAs_C::Dop() const {
    return ( NA );
}

MI_PROPERTY(InAs_C, cond, // TODO
            MIComment("no temperature dependence")
            )
Tensor2<double> InAs_C::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType InAs_C::condtype() const { return Material::CONDUCTIVITY_P; }

bool InAs_C::isEqual(const Material &other) const {
    const InAs_C& o = static_cast<const InAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAs::isEqual(other);
}

static MaterialsDB::Register<InAs_C> materialDB_register_InAs_C;

}} // namespace plask::materials
