#include "InP_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InP_Si::name() const { return NAME; }

std::string InP_Si::str() const { return StringBuilder("InP").dopant("Si", ND); }

InP_Si::InP_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 3900e-4/(1+pow((Nf_RT/1e18),0.51)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(InP_Si, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP_Si::mob(double T) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InP_Si, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Si::Nf(double T) const {
    return ( Nf_RT );
}

double InP_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InP_Si, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> InP_Si::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return (Tensor2<double>(tCond, tCond));
}

MI_PROPERTY(InP_Si, absp,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Si::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 1.6 * pow(Nf_RT/1e18, 0.94);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 0.7 * pow(Nf_RT/1e18, 1.14);
    else if ((wl > 2230.) && (wl < 2430.)) // only for 2330 nm TODO
        tAbsp = 2. * (Nf_RT/1e18);
    else if ((wl > 8900.) && (wl < 9100.)) // only for 9000 nm TODO
        tAbsp = 58. * (Nf_RT/1e18);
    return ( tAbsp );
}

static MaterialsDB::Register<InP_Si> materialDB_register_InP_Si;

} // namespace plask
