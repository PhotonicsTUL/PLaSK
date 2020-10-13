#include "GaInAs_Zn.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaInAs_Zn::name() const { return NAME; }

std::string GaInAs_Zn::str() const { return StringBuilder("In", In)("Ga")("As").dopant("Zn", NA); }

MI_PARENT(GaInAs_Zn, GaInAs)

GaInAs_Zn::GaInAs_Zn(const Material::Composition& Comp, double Val): GaInAs(Comp)/*, mGaAs_Zn(Val), mInAs_Zn(Val)*/
{
    if (In == 0.53) Nf_RT = 0.90*Val;
    else Nf_RT = Val;
    NA = Val;
    if (In == 0.53)
        mob_RT = 250./(1+pow((Nf_RT/6e17),0.34));
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_Zn, mob,
            MISource("TODO"),
            MISource("based on Zn-doped GaInAs")
            )
Tensor2<double> GaInAs_Zn::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_Zn, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaInAs_Zn::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double GaInAs_Zn::doping() const {
    return ( NA );
}

MI_PROPERTY(GaInAs_Zn, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs_Zn::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType GaInAs_Zn::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(GaInAs_Zn, absp,
            MISource("fit to ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaInAs_Zn::absp(double lam, double /*T*/) const {
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

bool GaInAs_Zn::isEqual(const Material &other) const {
    const GaInAs_Zn& o = static_cast<const GaInAs_Zn&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaInAs::isEqual(other);
}

static MaterialsDB::Register<GaInAs_Zn> materialDB_register_GaInAs_Zn;

}} // namespace plask::materials
