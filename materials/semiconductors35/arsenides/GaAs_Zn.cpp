#include "GaAs_Zn.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaAs_Zn::name() const { return NAME; }

std::string GaAs_Zn::str() const { return StringBuilder("GaAs").dopant("Zn", NA); }

GaAs_Zn::GaAs_Zn(DopingAmountType Type, double Val) {
    Nf_RT = Val; // TODO (it is not from publication)
    NA = Val; // TODO (it is not from publication)
    mob_RT = 480./(1+pow((Nf_RT/4e17),0.47)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaAs_Zn, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Zn::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.25);
    return ( Tensor2<double>(mob_T,mob_T) );
}

MI_PROPERTY(GaAs_Zn, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaAs_Zn::Nf(double T) const {
    return ( Nf_RT );
}

double GaAs_Zn::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaAs_Zn, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Zn::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType GaAs_Zn::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(GaAs_Zn, absp,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaAs_Zn::absp(double wl, double T) const {
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

bool GaAs_Zn::isEqual(const Material &other) const {
    const GaAs_Zn& o = static_cast<const GaAs_Zn&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaAs::isEqual(other);
}

static MaterialsDB::Register<GaAs_Zn> materialDB_register_GaAs_Zn;

}} // namespace plask::materials
