#include "InN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

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
double InN_Si::mob(double T) const {
    return ( mob_RT*(T*T*5.174E-6 -T*5.241E-3 +2.107) );
}

MI_PROPERTY(InN_Si, Nf,
            MISource("E. S. Hwang et al., J. Korean Phys. Soc. 48 (2006) 93"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("Si: 6e17 - 7e18 cm^-3")
            )
double InN_Si::Nf(double T) const {
	return ( Nf_RT*(-T*T*3.802E-6 +T*3.819E-3 +0.1965) );
}

double InN_Si::Dop() const {
    return ND;
}

MI_PROPERTY(InN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
double InN_Si::cond(double T) const {
	return ( 1.602E-17*Nf(T)*mob(T) );
}

MaterialsDB::Register<InN_Si> materialDB_register_InN_Si;

}       // namespace plask
