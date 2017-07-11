#include "InN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PARENT(InN_Si, InN)

std::string InN_Si::name() const { return NAME; }

std::string InN_Si::str() const { return StringBuilder("InN").dopant("Si", ND); }

InN_Si::InN_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 2.753e13*pow(Nf_RT,-0.559);
}

MI_PROPERTY(InN_Si, mob,
            MISource("E. S. Hwang et al., J. Korean Phys. Soc. 48 (2006) 93"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("based on 6 papers (2005-2010): undoped/Si-doped InN/c-sapphire")
            )
Tensor2<double> InN_Si::mob(double T) const {
    double tMob = mob_RT*(T*T*5.174E-6 -T*5.241E-3 +2.107);
    return (Tensor2<double>(tMob, tMob));
}

MI_PROPERTY(InN_Si, Nf,
            MISource("E. S. Hwang et al., J. Korean Phys. Soc. 48 (2006) 93"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("Si: 6e17 - 7e18 cm^-3")
            )
double InN_Si::Nf(double T) const {
	return ( Nf_RT*(-T*T*3.802E-6 +T*3.819E-3 +0.1965) );
}

MI_PROPERTY(InN_Si, Na,
            MIComment("-")
            )
double InN_Si::Na() const {
    return ( 0. );
}

MI_PROPERTY(InN_Si, Nd,
            MIComment("-")
            )
double InN_Si::Nd() const {
    return ( ND );
}

double InN_Si::Dop() const {
    return ND;
}

MI_PROPERTY(InN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
Tensor2<double> InN_Si::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

Material::ConductivityType InN_Si::condtype() const { return Material::CONDUCTIVITY_N; }

bool InN_Si::isEqual(const Material &other) const {
    const InN_Si& o = static_cast<const InN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InN::isEqual(other);
}

MaterialsDB::Register<InN_Si> materialDB_register_InN_Si;

}}       // namespace plask::materials
