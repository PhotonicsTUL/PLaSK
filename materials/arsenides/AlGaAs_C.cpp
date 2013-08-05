#include "AlGaAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaAs_C::name() const { return NAME; }

std::string AlGaAs_C::str() const { return StringBuilder("Al", Al)("Ga")("As").dopant("C", NA); }

MI_PARENT(AlGaAs_C, AlGaAs)

AlGaAs_C::AlGaAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaAs(Comp), mGaAs_C(Type,Val), mAlAs_C(Type,Val)
{
    //double act_GaAs = 0.92;
    //double fx1 = 1.;
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = mGaAs_C.Dop(); // mGaAs_C.Dop()*fx1;
    }
    else {
        Nf_RT = mGaAs_C.Nf(300.); // mGaAs_C.Nf(300.)*fx1;
        NA = Val;
    }
    double fx2 = 0.66 / (1. + pow(Al/0.21,3.)) + 0.34; // (1.00-0.34) / (1. + pow(Al/0.21,3.)) + 0.34;
    mob_RT = mGaAs_C.mob(300.).c00 * fx2;
}

MI_PROPERTY(AlGaAs_C, mob,
            MISource("based on 4 papers 1991-2000 about C-doped AlGaAs"),
            MISource("based on C-doped GaAs")
            )
Tensor2<double> AlGaAs_C::mob(double T) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(AlGaAs_C, Nf,
            MISource("based on 3 papers 1991-2004 about C-doped AlGaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs_C::Nf(double T) const {
    return ( Nf_RT );
}

double AlGaAs_C::Dop() const {
    return ( NA );
}

Tensor2<double> AlGaAs_C::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

bool AlGaAs_C::isEqual(const Material &other) const {
    const AlGaAs_C& o = static_cast<const AlGaAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlGaAs::isEqual(other);
}

static MaterialsDB::Register<AlGaAs_C> materialDB_register_AlGaAs_C;

}} // namespace plask::materials
