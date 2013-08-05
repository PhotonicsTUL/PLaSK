#include "GaInAs_Sn.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaInAs_Sn::name() const { return NAME; }

std::string GaInAs_Sn::str() const { return StringBuilder("Ga")("In", In)("As").dopant("Sn", ND); }

MI_PARENT(GaInAs_Sn, GaInAs)

GaInAs_Sn::GaInAs_Sn(const Material::Composition& Comp, DopingAmountType Type, double Val): GaInAs(Comp)/*, mGaAs_Sn(Type,Val), mInAs_Sn(Type,Val)*/
{
    Nf_RT = Val; // TODO
    ND = Val; // TODO

    if (In == 0.53)
        mob_RT = 10600e-4/(1+pow((Nf_RT/2e17),0.39)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    else
        mob_RT = 0.; // TODO
}

MI_PROPERTY(GaInAs_Sn, mob,
            MISource("TODO"),
            MISource("based on Si-doped GaInAs")
            )
Tensor2<double> GaInAs_Sn::mob(double T) const {
    return ( Tensor2<double>(mob_RT, mob_RT) );
}

MI_PROPERTY(GaInAs_Sn, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaInAs_Sn::Nf(double T) const {
    return ( Nf_RT );
}

double GaInAs_Sn::Dop() const {
    return ( ND );
}

MI_PROPERTY(GaInAs_Sn, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs_Sn::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaInAs_Sn, absp,
            MISource("fit to ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaInAs_Sn::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 18600. * pow(Nf_RT/1e18-3.1, -0.64);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 7600. * pow(Nf_RT/1e18, -2.0);
    return ( tAbsp );
}

bool GaInAs_Sn::isEqual(const Material &other) const {
    const GaInAs_Sn& o = static_cast<const GaInAs_Sn&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaInAs::isEqual(other);
}

static MaterialsDB::Register<GaInAs_Sn> materialDB_register_GaInAs_Sn;

}} // namespace plask::materials
