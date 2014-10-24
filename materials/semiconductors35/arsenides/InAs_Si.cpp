#include "InAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAs_Si::name() const { return NAME; }

std::string InAs_Si::str() const { return StringBuilder("InAs").dopant("Si", ND); }

InAs_Si::InAs_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val; //TODO
    ND = Val; //TODO
    mob_RT = 15000e-4/(1.+pow((Nf_RT/1e18),0.81)); // 1e-4: cm^2/(V*s) -> m^2/(V*s);
}

MI_PROPERTY(InAs_Si, mob,
            MISource("L.-G. Li, Chin. Phys. Lett. 29 (2012) pp. 076801"),
            MIComment("mob(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAs_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,0.8);
    return ( Tensor2<double>(tmob,tmob) );
}

MI_PROPERTY(InAs_Si, Nf,
            MIComment("Nf(ND) assumed, TODO: find exp. data"),
            MIComment("no temperature dependence")
            )
double InAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double InAs_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InAs_Si, cond,
            MIComment("cond(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAs_Si::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT;
    double tCond = tCond_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

bool InAs_Si::isEqual(const Material &other) const {
    const InAs_Si& o = static_cast<const InAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAs::isEqual(other);
}

static MaterialsDB::Register<InAs_Si> materialDB_register_InAs_Si;

}} // namespace plask::materials
