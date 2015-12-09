#include "AlN_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlN_Si::name() const { return NAME; }

std::string AlN_Si::str() const { return StringBuilder("AlN").dopant("Si", ND); }

MI_PARENT(AlN_Si, AlN)

AlN_Si::AlN_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIERS_CONCENTRATION) {
        Nf_RT = Val;
        ND = 1.223e10*pow(Val,5.540e-1);
    }
    else {
        Nf_RT = 6.197E-19*pow(Val,1.805);
        ND = Val;
    }
    //mobRT(Nf_RT),
    mob_RT = 29.410*exp(-1.838E-17*Nf_RT);
}

MI_PROPERTY(AlN_Si, mob,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MIComment("based on 4 papers (2004-2008): Si-doped AlN")
			)
Tensor2<double> AlN_Si::mob(double T) const {
    double tMob = mob_RT * (1.486 - T*0.00162);
    return (Tensor2<double>(tMob,tMob));
}

MI_PROPERTY(AlN_Si, Nf,
            MISource("Y. Taniyasu, Nature Letters 44 (2006) 325"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("based on 2 papers (2004-2008): Si-doped AlN")
            )
double AlN_Si::Nf(double T) const {
	return ( Nf_RT * 3.502E-27*pow(T,10.680) );
}

double AlN_Si::Dop() const {
    return ND;
}

MI_PROPERTY(AlN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
Tensor2<double> AlN_Si::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

MI_PROPERTY(AlN_Si, absp,
            MISeeClass<AlN>(MaterialInfo::absp)
            )
double AlN_Si::absp(double wl, double T) const {
    double a = phys::h_eVc1e9/wl - 6.28,
           b = ND/1e18;
    return ( (19000+400*b)*exp(a/(0.019+0.001*b)) + (330+200*b)*exp(a/(0.07+0.016*b)) );
}

bool AlN_Si::isEqual(const Material &other) const {
    const AlN_Si& o = static_cast<const AlN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<AlN_Si> materialDB_register_AlN_Si;

}}       // namespace plask::materials
