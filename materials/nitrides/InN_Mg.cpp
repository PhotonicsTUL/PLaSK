#include "InN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PARENT(InN_Mg, InN)

std::string InN_Mg::name() const { return NAME; }

std::string InN_Mg::str() const { return StringBuilder("InN").dopant("Mg", NA); }

InN_Mg::InN_Mg(DopingAmountType Type, double Val) {
    if (Type == CARRIERS_CONCENTRATION) {
        Nf_RT = Val;
        NA = 7.392E9*pow(Val,0.439);
    }
    else {
        Nf_RT = 3.311E-23*pow(Val,2.278);
        NA = Val;
    }
    mob_RT = 5.739E13*pow(Nf_RT,-0.663);
    cond_RT = phys::qe*100.*Nf_RT*mob_RT;
}

MI_PROPERTY(InN_Mg, mob,
            MISource("based on 4 papers (2006-2010): MBE-grown Mg-doped InN"),
            MIComment("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370")
            )
Tensor2<double> InN_Mg::mob(double T) const {
    return (Tensor2<double>(mob_RT,mob_RT));
}

MI_PROPERTY(InN_Mg, Nf,
            MISource("based on 2 papers (2008-2009): Mg-doped InN"),
            MIComment("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370")
            )
double InN_Mg::Nf(double T) const {
    return ( Nf_RT );
}

double InN_Mg::Dop() const {
    return NA;
}

MI_PROPERTY(InN_Mg, cond,
            MIComment("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370")
            )
Tensor2<double> InN_Mg::cond(double T) const {
    return (Tensor2<double>(cond_RT,cond_RT));
}

bool InN_Mg::isEqual(const Material &other) const {
    const InN_Mg& o = static_cast<const InN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && o.cond_RT == this->cond_RT && InN::isEqual(other);
}

MaterialsDB::Register<InN_Mg> materialDB_register_InN_Mg;

}}       // namespace plask::materials
