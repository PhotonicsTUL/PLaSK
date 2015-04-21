#include "GaN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register


namespace plask { namespace materials {

MI_PARENT(GaN_Si, GaN)

std::string GaN_Si::name() const { return NAME; }

std::string GaN_Si::str() const { return StringBuilder("GaN").dopant("Si", ND); }

GaN_Si::GaN_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        ND = std::pow(Val/0.55,1/1.01);
    }
    else {
        Nf_RT = 0.55*std::pow(Val,1.01);
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
    double tMob = mob_RT*(1.486-T*0.001619);
    return (Tensor2<double>(tMob,tMob));
}

MI_PROPERTY(GaN_Si, Nf,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("Y. Oshima et al., Phys. Status Solidi C 4 (2007) 2215"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MIComment("In the RT Nf(ND) for Si: 6e17 - 7e18 cm^-3")
            )
double GaN_Si::Nf(double T) const {
    return ( Nf_RT*(0.638+T*0.001217) );
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
            MIComment("Nf: 1e18 - 2e19 cm^-3")
            )
Tensor2<double> GaN_Si::thermk(double T, double t) const {
    double fun_Nf = std::exp(-4.67*Nf_RT/1e21);
    auto p = GaN::thermk(T,t);
    p.c00 *= fun_Nf;
    p.c11 *= fun_Nf;
    return p;
 }

MI_PROPERTY(GaN_Si, absp,
            MISource("P. Perlin et al., SPIE 8262, 826216"),
            MIArgumentRange(MaterialInfo::wl, 380, 500),
            MIComment("more data: 380, 390, 400, 420, 430, 440, 450"),
            MIComment("GaN:Si if Nf > 5e18 cm-3, else GaN(undoped)"),
            MIComment("no temperature dependence")
            )
double GaN_Si::absp(double wl, double T) const {
    double absp_t = 0.;
    if (Nf(T) > 5e18) absp_t = (33500*exp(0.8*Nf(T)/1e19))*exp(wl*(-0.0018*Nf(T)/1e19-0.0135));
    else absp_t = GaN::absp(wl,T);
    return absp_t;
}

bool GaN_Si::isEqual(const Material &other) const {
    const GaN_Si& o = static_cast<const GaN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

MI_PROPERTY(GaN_Si, nr,
            MISeeClass<GaN>(MaterialInfo::nr),
            MISource("P. Perlin et al., SPIE 8262, 826216"),
            MIComment("Nf > 1e19 cm-3"),
            MIComment("no temperature dependence")
            )
double GaN_Si::nr(double wl, double T, double n) const {
    return ( GaN::nr(wl,T) * (1-1.05*Nf_RT/1e22) );
}


Tensor2<double> GaN_Si_bulk::thermk(double T, double t) const {
    return GaN_Si::thermk(T, INFINITY);
}


static MaterialsDB::Register<GaN_Si> materialDB_register_GaN_Si;

static MaterialsDB::Register<GaN_Si_bulk> materialDB_register_GaN_Si_bulk;

}}       // namespace plask::materials
