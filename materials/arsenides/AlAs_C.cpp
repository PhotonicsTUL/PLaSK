#include "AlAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlAs_C::name() const { return NAME; }

std::string AlAs_C::str() const { return StringBuilder("AlAs").dopant("C", NA); }

AlAs_C::AlAs_C(DopingAmountType Type, double Val) {
    //double act_GaAs = 0.92;
    //double fx1 = 1.;
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.92; // Val/(act_GaAs*fx1);
    }
    else {
        Nf_RT = 0.92*Val; // (act_GaAs*fx1)*Val;
        NA = Val;
    }
    double mob_RT_GaAs = 530e-4/(1+pow((Nf_RT/1e17),0.30)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    //double Al = 1.; // AlAs (not AlGaAs)
    double fx2 = 0.66 / (1. + pow(1./0.21,3.)) + 0.34; // (1.00-0.34) / (1. + pow(Al/0.21,3.)) + 0.34;
    mob_RT = mob_RT_GaAs * fx2;
}

MI_PROPERTY(AlAs_C, mob,
            MIComment("TODO")
            )
Tensor2<double> AlAs_C::mob(double T) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(AlAs_C, Nf,
            MIComment("TODO")
            )
double AlAs_C::Nf(double T) const {
    return ( Nf_RT );
}

double AlAs_C::Dop() const {
    return ( NA );
}

MI_PROPERTY(AlAs_C, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs_C::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

bool AlAs_C::isEqual(const Material &other) const {
    const AlAs_C& o = static_cast<const AlAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAs::isEqual(other);
}

static MaterialsDB::Register<AlAs_C> materialDB_register_AlAs_C;

} // namespace plask
