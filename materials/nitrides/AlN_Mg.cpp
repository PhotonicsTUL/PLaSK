#include "AlN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlN_Mg::name() const { return NAME; }

std::string AlN_Mg::str() const { return StringBuilder("AlN").dopant("Mg", NA); }

MI_PARENT(AlN_Mg, AlN)

AlN_Mg::AlN_Mg(DopingAmountType Type, double Val) {
    Nf_RT = 2e11;
    mob_RT = 10.;
    cond_RT = 3e-5;
    NA = 2e19;
}

Tensor2<double> AlN_Mg::mob(double T) const {
    return Tensor2<double>(mob_RT, mob_RT);
}

double AlN_Mg::Nf(double T) const {
    return Nf_RT;
}


double AlN_Mg::Dop() const {
    return NA;
}


MI_PROPERTY(AlN_Mg, cond,
            MISource("K. B. Nam et al., Appl. Phys. Lett. 83 (2003) 878"),
            MISource("M. L. Nakarmi et al., Appl. Phys. Lett. 89 (2006) 152120"),
            MIArgumentRange(MaterialInfo::T, 300, 900)
            )
Tensor2<double> AlN_Mg::cond(double T) const {
    double tCond = 3e-5*pow((T/300.),9.75);
    return (Tensor2<double>(tCond, tCond)); //TODO was Tensor2<double>(tCondt, tCond) - compilation error
}

Material::ConductivityType AlN_Mg::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(AlN_Mg, absp,
            MISeeClass<AlN>(MaterialInfo::absp)
            )
double AlN_Mg::absp(double wl, double T) const {
    double a = phys::h_eVc1e9/wl - 6.28,
           b = NA/1e18;
    return ( (19000+200*b)*exp(a/(0.019+0.0001*b)) + (330+30*b)*exp(a/(0.07+0.0008*b)) );
}

bool AlN_Mg::isEqual(const Material &other) const {
    const AlN_Mg& o = static_cast<const AlN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && o.cond_RT == this->cond_RT;
}

static MaterialsDB::Register<AlN_Mg> materialDB_register_AlN_Mg;

}}       // namespace plask::materials
