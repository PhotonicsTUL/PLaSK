#include "GaAs_Si.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaAs_Si::name() const { return NAME; }

std::string GaAs_Si::str() const { return StringBuilder("GaAs").dopant("Si", ND); }

GaAs_Si::GaAs_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 6600e-4/(1+pow((Nf_RT/5e17),0.53)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaAs_Si, mob,
            MISource("fit to n-GaAs:Si (based on 8 papers 1982 - 2003)"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Si::mob(double T) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(GaAs_Si, Nf,
            MISource("based on 3 papers 1982 - 1996"),
            MIComment("no temperature dependence")
            )
double GaAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double GaAs_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(GaAs_Si, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_Si::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaAs_Si, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double GaAs_Si::absp(double wl, double T) const {
    double tDWl = 1240.*(Eg(300.,0.,'G')-Eg(T,0.,'G'))/(Eg(300.,0.,'G')*Eg(T,0.,'G'));
    double tWl = (wl-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6000.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0169)+4.67+0.00211*pow(tWl,4.80));
    else if (tWl <= 27000.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(-8.4+0.233*pow(tWl,2.6));
    return ( tAbsp );
}

bool GaAs_Si::isEqual(const Material &other) const {
    const GaAs_Si& o = static_cast<const GaAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaAs::isEqual(other);
}

static MaterialsDB::Register<GaAs_Si> materialDB_register_GaAs_Si;

}} // namespace plask::materials
