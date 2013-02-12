#include "GaInAs_Be.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaInAs_Be::name() const { return NAME; }

std::string GaInAs_Be::str() const { return StringBuilder("Ga")("In", In)("As").dopant("Be", NA); }

MI_PARENT(GaInAs_Be, GaInAs)

GaInAs_Be::GaInAs_Be(const Material::Composition& Comp, DopingAmountType Type, double Val): GaInAs(Comp)/*, mGaAs_Be(Type,Val), mInAs_Be(Type,Val)*/
{
    Nf_RT = Val; // TODO
    NA = Val; // TODO
    if (In == 0.53)
        mob_RT = 120e-4/(1+pow((Nf_RT/2e19),0.39)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_Be, mob,
            MISource("TODO"),
            MISource("based on Be-doped GaInAs")
            )
Tensor2<double> GaInAs_Be::mob(double T) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_Be, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaInAs_Be::Nf(double T) const {
    return ( Nf_RT );
}

double GaInAs_Be::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaInAs_Be, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs_Be::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaInAs_Be, absp,
            MISource("fit to ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaInAs_Be::absp(double wl, double T) const {
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

bool GaInAs_Be::isEqual(const Material &other) const {
    const GaInAs_Be& o = static_cast<const GaInAs_Be&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<GaInAs_Be> materialDB_register_GaInAs_Be;

} // namespace plask
