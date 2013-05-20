#include "GaN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register


namespace plask {

MI_PARENT(GaN_Si, GaN)

std::string GaN_Si::name() const { return NAME; }

std::string GaN_Si::str() const { return StringBuilder("GaN").dopant("Si", ND); }

GaN_Si::GaN_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        ND = 5.905*pow(Val,0.962);
    }
    else {
        Nf_RT = 0.158*pow(Val,1.039);
        ND = Val;
    }
    mob_RT = 4.164e6*pow(Nf_RT,-0.228);
}

MI_PROPERTY(GaN_Si, mob,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MIComment("based on 7 papers (1996-2007): undoped/Si-doped GaN/c-sapphire")
            )
Tensor2<double> GaN_Si::mob(double T) const {
    double tMob = mob_RT*(1.486-T*0.00162);
    return (Tensor2<double>(tMob,tMob));
}

MI_PROPERTY(GaN_Si, Nf,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("Y. Oshima et al., Phys. Status Solidi C 4 (2007) 2215"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MIComment("Si: 6e17 - 7e18 cm^-3")
            )
double GaN_Si::Nf(double T) const {
	return ( Nf_RT*(-T*T*3.405e-6 + T*3.553e-3 + 0.241) );
}

double GaN_Si::Dop() const {
    return ND;
}

MI_PROPERTY(GaN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
Tensor2<double> GaN_Si::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

MI_PROPERTY(GaN_Si, thermk,
            MISeeClass<GaN>(MaterialInfo::thermk),
            MISource("Y. Oshima et al., Phys. Status Solidi C 4 (2007) 2215"),
            MIComment("Nf: 1e18 - 1e19 cm^-3")
            )
Tensor2<double> GaN_Si::thermk(double T, double t) const {
    double fun_Nf = 2.18*std::pow(Nf_RT,-0.022);
    auto p = GaN::thermk(T,t);
    p.c00 *= fun_Nf;
    p.c11 *= fun_Nf;
    return p;
 }

MI_PROPERTY(GaN_Si, absp,
            MISource("Perlin Unipress 11.2011 unpublished"),
            MIArgumentRange(MaterialInfo::wl, 410, 410),
            MIComment("more data: 380, 390, 400, 420, 430, 440, 450"),
            MIComment("Nf: 1e18 - 5e19 cm-3"),
            MIComment("no temperature dependence")
            )
double GaN_Si::absp(double wl, double T) const {
    return ( (33479.3*exp(0.0804278*Nf(T)/1e18))*exp((-0.000180411*Nf(T)-0.0135216)*wl) );
}

bool GaN_Si::isEqual(const Material &other) const {
    const GaN_Si& o = static_cast<const GaN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

MI_PROPERTY(GaN_Si, nr,
            MISeeClass<GaN>(MaterialInfo::nr),
            MISource("Perlin Unipress 11.2011 unpublished"),
            MIComment("Nf: 1e18 - 5e19 cm-3"),
            MIComment("no temperature dependence")
            )
double GaN_Si::nr(double wl, double T) const {
	return ( GaN::nr(wl,T) * (1.0001-Nf_RT/1e18*1.05003e-4 ) );
}

MaterialsDB::Register<GaN_Si> materialDB_register_GaN_Si;

}       // namespace plask
