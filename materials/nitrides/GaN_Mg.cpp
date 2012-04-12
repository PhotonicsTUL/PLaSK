#include "GaN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register


namespace plask {

MI_PARENT(GaN_Mg, GaN)

GaN_Mg::GaN_Mg(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = 1.561e-3*pow(Val,1.262);
    }
    else {
        Nf_RT = 1.676E2*pow(Val,0.7925);
        NA = Val;
    }
    mob_RT = 25.747*exp(-9.034E-19*Nf_RT);
}

std::string GaN_Mg::name() const { return NAME; }


MI_PROPERTY(GaN_Mg, mob,
            MISource("P. Kozodoy et al., J. Appl. Phys. 87 (2000) 1832"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("based on 9 papers (2000-2009): MBE-grown Mg-doped GaN"),
            MIComment("Nf: 2e17 - 6e18 cm^-3")
            )
double GaN_Mg::mob(double T) const {
    return ( mob_RT * (T*T*2.495E-5 -T*2.268E-2 +5.557) );
}

MI_PROPERTY(GaN_Mg, Nf,
            MISource("P. Kozodoy et al., J. Appl. Phys. 87 (2000) 1832"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("based on 4 papers (1998-2008): MBE-grown Mg-doped GaN"),
            MIComment("Mg: 1e19 - 8e20 cm^-3")
            )
double GaN_Mg::Nf(double T) const {
	return ( Nf_RT * (T*T*2.884E-4 -T*0.147 + 19.080) );
}

double GaN_Mg::Dop() const {
    return NA;
}

MI_PROPERTY(GaN_Mg, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
double GaN_Mg::cond(double T) const {
	return ( 1.602E-17*Nf(T)*mob(T) );
}

MI_PROPERTY(GaN_Mg, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("no temperature dependence")
            )
double GaN_Mg::absp(double wl, double T) const {
    double a = 1239.84190820754/wl - 3.42;
    return ( (19000+200*NA)*exp(a/(0.019+0.0001*NA)) + (330+30*NA)*exp(a/(0.07+0.000*NA)) );
}

static MaterialsDB::Register<GaN_Mg> materialDB_register_Mg;

}       // namespace plask
