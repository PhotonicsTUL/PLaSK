#include "GaAs_Be.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaAs_Be::name() const { return NAME; }

std::string GaAs_Be::str() const { return StringBuilder("GaAs").dopant("Be", NA); }

GaAs_Be::GaAs_Be(DopingAmountType Type, double Val) {
    Nf_RT = Val; // TODO (add source)
    NA = Val; // TODO (add source)
    mob_RT = 840./(1+pow((Nf_RT/1e16),0.28));
}

MI_PROPERTY(GaAs_Be, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Be::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.25);
    return (Tensor2<double>(mob_T,mob_T));
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
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
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

bool GaAs_Be::isEqual(const Material &other) const {
    const GaAs_Be& o = static_cast<const GaAs_Be&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaAs::isEqual(other);
}

static MaterialsDB::Register<GaAs_Be> materialDB_register_GaAs_Be;

}} // namespace plask::materials
