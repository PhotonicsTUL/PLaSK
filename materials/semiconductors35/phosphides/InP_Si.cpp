#include "InP_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InP_Si::name() const { return NAME; }

std::string InP_Si::str() const { return StringBuilder("InP").dopant("Si", ND); }

InP_Si::InP_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 3900./(1+pow((Nf_RT/1e18),0.51));
}

MI_PROPERTY(InP_Si, mob,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InP_Si::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InP_Si, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Si::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double InP_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InP_Si, cond,
            MIComment("")
            )
Tensor2<double> InP_Si::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT*1e-4 * pow(300./T,0.8);
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType InP_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(InP_Si, absp,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double InP_Si::absp(double lam, double /*T*/) const {
    double tAbsp(0.);
    if ((lam > 1200.) && (lam < 1400.)) // only for 1300 nm TODO
        tAbsp = 1.6 * pow(Nf_RT/1e18, 0.94);
    else if ((lam > 1450.) && (lam < 1650.)) // only for 1550 nm TODO
        tAbsp = 0.7 * pow(Nf_RT/1e18, 1.14);
    else if ((lam > 2230.) && (lam < 2430.)) // only for 2330 nm TODO
        tAbsp = 2. * (Nf_RT/1e18);
    else if ((lam > 8900.) && (lam < 9100.)) // only for 9000 nm TODO
        tAbsp = 58. * (Nf_RT/1e18);
    else if ((lam > 9200.) && (lam < 10000.)) { // only for about 9500 nm (based on www.ioffe.ru/SVA/NSM/Semicond/InP)
        double tEf = phys::PhotonEnergy(lam),
               tAbsp_n2e16 = 0.01435*pow(tEf,-2.5793),
               tAbsp_n2e17 = 0.04715*pow(tEf,-2.6173),
               tAbsp_n4e17 = 0.04331*pow(tEf,-3.0428);
        if (Nf_RT < 2e17) tAbsp = (tAbsp_n2e17-tAbsp_n2e16)*(Nf_RT-2e16)/1.8E17+tAbsp_n2e16;
        else tAbsp =(tAbsp_n4e17-tAbsp_n2e17)*(Nf_RT-2e17)/2E17+tAbsp_n2e17;
    }
    return ( tAbsp );
}

bool InP_Si::isEqual(const Material &other) const {
    const InP_Si& o = static_cast<const InP_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<InP_Si> materialDB_register_InP_Si;

}} // namespace plask::materials
