#include "GaInAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaInAs_C::name() const { return NAME; }

std::string GaInAs_C::str() const { return StringBuilder("In", In)("Ga")("As").dopant("C", NA); }

MI_PARENT(GaInAs_C, GaInAs)

GaInAs_C::GaInAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val): GaInAs(Comp)/*, mGaAs_C(Type,Val), mInAs_C(Type,Val)*/
{
    Nf_RT = Val; // TODO
    NA = Val; // TODO
    if (In == 0.53)
        mob_RT = 570e-4/(1+pow((Nf_RT/9e14),0.21)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_C, mob,
            MISource("TODO"),
            MISource("based on C-doped GaInAs")
            )
Tensor2<double> GaInAs_C::mob(double T) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_C, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaInAs_C::Nf(double T) const {
    return ( Nf_RT );
}

double GaInAs_C::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaInAs_C, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs_C::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaInAs_C, absp,
            MISource("fit to ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaInAs_C::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 60500. * pow(Nf_RT/1e18+23.3, -0.54);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 24000. * pow(Nf_RT/1e18+9.7, -0.61);
    else if ((wl > 2230.) && (wl < 2430.)) // only for 2330 nm TODO
        tAbsp = 63. * pow(Nf_RT/1e18, -0.7);
    else if ((wl > 8900.) && (wl < 9100.)) // only for 9000 nm TODO
        tAbsp = 250. * pow(Nf_RT/1e18, -0.7);
    return ( tAbsp );
}

bool GaInAs_C::isEqual(const Material &other) const {
    const GaInAs_C& o = static_cast<const GaInAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaInAs::isEqual(other);
}

static MaterialsDB::Register<GaInAs_C> materialDB_register_GaInAs_C;

}} // namespace plask::materials
