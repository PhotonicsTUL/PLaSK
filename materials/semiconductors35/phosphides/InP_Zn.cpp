#include "InP_Zn.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InP_Zn::name() const { return NAME; }

std::string InP_Zn::str() const { return StringBuilder("InP").dopant("Zn", NA); }

InP_Zn::InP_Zn(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.75;
    }
    else {
        Nf_RT = 0.75*Val;
        NA = Val;
    }
    mob_RT = 120./(1+pow((Nf_RT/2e18),1.00));
}

MI_PROPERTY(InP_Zn, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP_Zn::mob(double T) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InP_Zn, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Zn::Nf(double T) const {
    return ( Nf_RT );
}

double InP_Zn::Dop() const {
    return ( NA );
}

MI_PROPERTY(InP_Zn, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> InP_Zn::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT*1e-4;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType InP_Zn::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(InP_Zn, absp,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Zn::absp(double wl, double T) const {
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

bool InP_Zn::isEqual(const Material &other) const {
    const InP_Zn& o = static_cast<const InP_Zn&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<InP_Zn> materialDB_register_InP_Zn;

}} // namespace plask::materials
