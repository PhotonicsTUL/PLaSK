#include "GaInAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaInAs_Si::name() const { return NAME; }

std::string GaInAs_Si::str() const { return StringBuilder("In", In)("Ga")("As").dopant("Si", ND); }

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
        mob_RT = 16700./(1+pow((Nf_RT/6e16),0.42));
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
            MIComment("")
            )
Tensor2<double> GaInAs_Si::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4 * pow(300./T,1.59);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaInAs_Si, absp,
            MISource("calculations of the absorption for 9.5 um are based on http://www.ioffe.ru/SVA/NSM/Semicond"), // TODO
            MIComment("temperature dependence only for the wavelength of about 9.5 um")
            )
double GaInAs_Si::absp(double wl, double T) const {
    double tAbsp(0.);
    if ((wl > 1200.) && (wl < 1400.)) // only for 1300 nm TODO
        tAbsp = 18600. * pow(Nf_RT/1e18-3.1, -0.64);
    else if ((wl > 1450.) && (wl < 1650.)) // only for 1550 nm TODO
        tAbsp = 7600. * pow(Nf_RT/1e18, -2.0);
    else if ((wl > 9000.) && (wl < 10000.)) // only for about 9.5 um TODO
    {
        double tNf = Nf_RT/1e16;
        tAbsp = ((0.00086*tNf*tNf+0.3*tNf+0.74)*pow(Ga,0.0012*tNf+0.025))*(1+0.001289*(T-300));
    }
    return ( tAbsp );
}

bool GaInAs_Si::isEqual(const Material &other) const {
    const GaInAs_Si& o = static_cast<const GaInAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaInAs::isEqual(other);
}

static MaterialsDB::Register<GaInAs_Si> materialDB_register_GaInAs_Si;

}} // namespace plask::materials
