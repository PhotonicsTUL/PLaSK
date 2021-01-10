#include "GaInAs_Be.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaInAs_Be::name() const { return NAME; }

std::string GaInAs_Be::str() const { return StringBuilder("In", In)("Ga")("As").dopant("Be", NA); }

MI_PARENT(GaInAs_Be, GaInAs)

GaInAs_Be::GaInAs_Be(const Material::Composition& Comp, double Val): GaInAs(Comp)/*, mGaAs_Be(Val), mInAs_Be(Val)*/
{
    Nf_RT = Val; // TODO (add source)
    NA = Val; // TODO (add source)
    if (In == 0.53)
        mob_RT = 120./(1+pow((Nf_RT/2e19),0.39));
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_Be, mob,
            MISource("TODO"),
            MISource("based on Be-doped GaInAs")
            )
Tensor2<double> GaInAs_Be::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_Be, Nf,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
double GaInAs_Be::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double GaInAs_Be::doping() const {
    return ( NA );
}

MI_PROPERTY(GaInAs_Be, cond,
            MINote("no temperature dependence")
            )
Tensor2<double> GaInAs_Be::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType GaInAs_Be::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(GaInAs_Be, absp,
            MISource("fit to ..."), // TODO
            MINote("no temperature dependence")
            )
double GaInAs_Be::absp(double lam, double /*T*/) const {
    double tAbsp(0.);
    if ((lam > 1200.) && (lam < 1400.)) // only for 1300 nm TODO
        tAbsp = 60500. * pow(Nf_RT/1e18+23.3, -0.54);
    else if ((lam > 1450.) && (lam < 1650.)) // only for 1550 nm TODO
        tAbsp = 24000. * pow(Nf_RT/1e18+9.7, -0.61);
    else if ((lam > 2230.) && (lam < 2430.)) // only for 2330 nm TODO
        tAbsp = 63. * pow(Nf_RT/1e18, -0.7);
    else if ((lam > 8900.) && (lam < 9100.)) // only for 9000 nm TODO
        tAbsp = 250. * pow(Nf_RT/1e18, -0.7);
    return ( tAbsp );
}

bool GaInAs_Be::isEqual(const Material &other) const {
    const GaInAs_Be& o = static_cast<const GaInAs_Be&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaInAs::isEqual(other);
}

static MaterialsDB::Register<GaInAs_Be> materialDB_register_GaInAs_Be;

}} // namespace plask::materials
