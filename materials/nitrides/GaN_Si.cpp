#include "GaN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register


namespace plask { namespace materials {

MI_PARENT(GaN_Si, GaN)

std::string GaN_Si::name() const { return NAME; }

std::string GaN_Si::str() const { return StringBuilder("GaN").dopant("Si", ND); }

GaN_Si::GaN_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIERS_CONCENTRATION) {
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
    return Tensor2<double>(tMob,tMob);
}

MI_PROPERTY(GaN_Si, Nf,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("Y. Oshima et al., Phys. Status Solidi C 4 (2007) 2215"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MIComment("In the RT Nf(ND) for Si: 6e17 - 7e18 cm^-3")
            )
double GaN_Si::Nf(double T) const {
    return Nf_RT*(0.638+T*0.001217) ;
}

MI_PROPERTY(GaN_Si, Na,
            MIComment("-")
            )
double GaN_Si::Na() const {
    return ( 0. );
}

MI_PROPERTY(GaN_Si, Nd,
            MIComment("-")
            )
double GaN_Si::Nd() const {
    return ( ND );
}

double GaN_Si::Dop() const {
    return ND;
}

MI_PROPERTY(GaN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
Tensor2<double> GaN_Si::cond(double T) const {
    return Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11);
}

Material::ConductivityType GaN_Si::condtype() const { return Material::CONDUCTIVITY_N; }

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
            MIArgumentRange(MaterialInfo::lam, 380, 500),
            MIComment("more data: 380, 390, 400, 420, 430, 440, 450"),
            MIComment("GaN:Si if Nf > 5e18 cm-3, else GaN(undoped)"),
            MIComment("no temperature dependence")
            )
double GaN_Si::absp(double lam, double T) const {
    double dE = phys::h_eVc1e9 / lam - Eg(T); // dE = E - Eg
    double N = Dop() * 1e-18;

    double tNgr = -0.0003878*lam*lam + 0.3946*lam - 90.42;
    if (N > tNgr) { // Perlin
        double n = Nf(T) * 1e-18;
        return 33500. * exp(0.08*n + (-0.00018*n - 0.0135) * lam);
    } else // Piprek
        return (19000.+4000.*N) * exp(dE / (0.019 + 0.001*N)) + (330.+200.*N) * exp(dE/(0.07+0.016* N));
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
double GaN_Si::nr(double lam, double T, double n) const {
    return GaN::nr(lam,T) * (1. - 1.05e-22 * (n?n:Nf(T)));
}


Tensor2<double> GaN_Si_bulk::thermk(double T, double /*t*/) const {
    return GaN_Si::thermk(T, INFINITY);
}


static MaterialsDB::Register<GaN_Si> materialDB_register_GaN_Si;

static MaterialsDB::Register<GaN_Si_bulk> materialDB_register_GaN_Si_bulk;

}}       // namespace plask::materials
