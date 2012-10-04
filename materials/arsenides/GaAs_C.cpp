#include "GaAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaAs_C::name() const { return NAME; }

std::string GaAs_C::str() const { return StringBuilder("GaAs").dopant("C", NA); }

GaAs_C::GaAs_C(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.92;
    }
    else {
        Nf_RT = 0.92*Val;
        NA = Val;
    }
    mob_RT = 530e-4/(1+pow((Nf_RT/1e17),0.30)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaAs_C, mob,
            MISource("fit to p-GaAs:C (based on 23 papers 1988 - 2006)"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs_C::mob(double T) const {
    return (std::make_pair(mob_RT,mob_RT));
}

MI_PROPERTY(GaAs_C, cond,
			MIComment("no temperature dependence")
            )
std::pair<double,double> GaAs_C::cond(double T) const {
    double tCond = cPhys::q * Nf_RT*1e6 * mob_RT;
    return (std::make_pair(tCond, tCond));
}

MI_PROPERTY(GaAs_C, absp,
            MISource("fit to ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaAs_C::absp(double wl, double T) const {
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

static MaterialsDB::Register<GaAs_C> materialDB_register_GaAs_C;

} // namespace plask
