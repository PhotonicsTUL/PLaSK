#include "InN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

MI_PARENT(InN_Mg, InN)

std::string InN_Mg::name() const { return NAME; }

std::string InN_Mg::str() const { return StringBuilder("InN").dopant("Mg", NA); }

InN_Mg::InN_Mg(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = 7.392E9*pow(Val,0.439);
    }
    else {
        Nf_RT = 3.311E-23*pow(Val,2.278);
        NA = Val;
    }
    mob_RT = 5.739E13*pow(Nf_RT,-0.663);
    cond_RT = 1.602E-17*Nf_RT*mob_RT;
}

MI_PROPERTY(InN_Mg, mob,
            MISource("based on 4 papers (2006-2010): MBE-grown Mg-doped InN"),
            MIComment("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370")
            )
std::pair<double,double> InN_Mg::mob(double T) const {
    return (std::make_pair(mob_RT,mob_RT));
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
std::pair<double,double> InN_Mg::cond(double T) const {
    return (std::make_pair(cond_RT,cond_RT));
}

MaterialsDB::Register<InN_Mg> materialDB_register_InN_Mg;

}       // namespace plask
