#include "GaAs_Be.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaAs_Be::name() const { return NAME; }

std::string GaAs_Be::str() const { return StringBuilder("GaAs").dopant("Be", NA); }

GaAs_Be::GaAs_Be(DopingAmountType Type, double Val) {
    Nf_RT = Val; // TODO (add source)
    NA = Val; // TODO (add source)
    mob_RT = 840e-4/(1+pow((Nf_RT/1e16),0.28)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaAs_Be, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Be::mob(double T) const {
    return (Tensor2<double>(mob_RT,mob_RT));
}

MI_PROPERTY(GaAs_Be, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaAs_Be::Nf(double T) const {
    return ( Nf_RT );
}

double GaAs_Be::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaAs_Be, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Be::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return (Tensor2<double>(tCond, tCond));
}

MI_PROPERTY(GaAs_Be, absp,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaAs_Be::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 9. * pow(Nf_RT/1e18, 1.33);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 25. * pow(Nf_RT/1e18, 1.1);
    else if ((wl > 2230.) && (wl < 2430.)) // only for 2330 nm TODO
        tAbsp = 320. * pow(Nf_RT/1e18, 0.7);
    else if ((wl > 8900.) && (wl < 9100.)) // only for 9000 nm TODO
        tAbsp = 1340. * pow(Nf_RT/1e18, 0.7);
    return ( tAbsp );
}

static MaterialsDB::Register<GaAs_Be> materialDB_register_GaAs_Be;

} // namespace plask
