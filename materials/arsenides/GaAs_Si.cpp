#include "GaAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaAs_Si::name() const { return NAME; }

std::string GaAs_Si::str() const { return StringBuilder("GaAs").dopant("Si", ND); }

GaAs_Si::GaAs_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 6600/(1+pow((Nf_RT/5e17),0.53));
}

MI_PROPERTY(GaAs_Si, mob,
            MISource("fit to n-GaAs:Si (based on 8 papers 1982 - 2003)"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs_Si::mob(double T) const {
    return (std::make_pair(mob_RT,mob_RT));
}

MI_PROPERTY(GaAs_Si, Nf,
            MISource("based on 3 papers 1982 - 1996"),
            MIComment("no temperature dependence")
            )
double GaAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double GaAs_Si::Dop() const {
    return ND;
}

MI_PROPERTY(GaAs_Si, cond,
			MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs_Si::cond(double T) const {
    double tCond = 1.602e-17*Nf_RT*mob_RT;
    return (std::make_pair(tCond, tCond));
}

MI_PROPERTY(GaAs_Si, absp,
            MISource(""),
            MIComment("TODO")
            )
double GaAs_Si::absp(double wl, double T) const {
    return ( 0 );
}


static MaterialsDB::Register<GaAs_Si> materialDB_register_GaAs_Si;

}       // namespace plask
