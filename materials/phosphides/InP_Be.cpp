#include "InP_Be.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string InP_Be::name() const { return NAME; }

std::string InP_Be::str() const { return StringBuilder("InP").dopant("Be", NA); }

InP_Be::InP_Be(DopingAmountType Type, double Val) {
    Nf_RT = Val; // TODO (it is not from publication)
    NA = Val; // TODO (it is not from publication)
    mob_RT = 140e-4/(1+pow((Nf_RT/1e18),0.50)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(InP_Be, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP_Be::mob(double T) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InP_Be, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Be::Nf(double T) const {
    return ( Nf_RT );
}

double InP_Be::Dop() const {
    return ( NA );
}

MI_PROPERTY(InP_Be, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> InP_Be::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return (Tensor2<double>(tCond, tCond));
}

MI_PROPERTY(InP_Be, absp,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Be::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 23. * pow(Nf_RT/1e18, 0.7);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 38. * pow(Nf_RT/1e18, 0.7);
    else if ((wl > 2230.) && (wl < 2430.)) // only for 2330 nm TODO
        tAbsp = 52. * pow(Nf_RT/1e18, 1.2);
    else if ((wl > 8900.) && (wl < 9100.)) // only for 9000 nm TODO
        tAbsp = 200. * (Nf_RT/1e18);
    return ( tAbsp );
}

static MaterialsDB::Register<InP_Be> materialDB_register_InP_Be;

} // namespace plask
