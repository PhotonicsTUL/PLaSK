#include "GaInAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string GaInAs_Si::name() const { return NAME; }

std::string GaInAs_Si::str() const { return StringBuilder("Ga")("In", In)("As").dopant("Si", ND); }

MI_PARENT(GaInAs_Si, GaInAs)

GaInAs_Si::GaInAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): GaInAs(Comp)/*, mGaAs_Si(Type,Val), mInAs_Si(Type,Val)*/
{
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        if (In == 0.53) ND = Val/0.55;
        else ND = Val;
    }
    else {
        if (In == 0.53) Nf_RT = 0.55*Val;
        else Nf_RT = Val;
        ND = Val;
    }
    if (In == 0.53)
        mob_RT = 16700e-4/(1+pow((Nf_RT/6e16),0.42)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_Si, mob,
            MISource("TODO"),
            MISource("based on Si-doped GaInAs")
            )
Tensor2<double> GaInAs_Si::mob(double T) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_Si, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaInAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double GaInAs_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(GaInAs_Si, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs_Si::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaInAs_Si, absp,
            MISource("fit to ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaInAs_Si::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 18600. * pow(Nf_RT/1e18-3.1, -0.64);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 7600. * pow(Nf_RT/1e18, -2.0);
    return ( tAbsp );
}

static MaterialsDB::Register<GaInAs_Si> materialDB_register_GaInAs_Si;

} // namespace plask
