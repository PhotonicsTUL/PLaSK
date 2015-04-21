#include "GaN_Mg.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register


namespace plask { namespace materials {

MI_PARENT(GaN_Mg, GaN)

std::string GaN_Mg::name() const { return NAME; }

std::string GaN_Mg::str() const { return StringBuilder("GaN").dopant("Mg", NA); }

GaN_Mg::GaN_Mg(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = std::pow(Val/0.65e4,1/0.71);
    }
    else {
        Nf_RT = 0.65E4*std::pow(Val,0.71);
        NA = Val;
    }
    mob_RT = 26.7*exp(-Nf_RT/1e18);
}

MI_PROPERTY(GaN_Mg, mob,
            MISource("P. Kozodoy et al., J. Appl. Phys. 87 (2000) 1832"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MIComment("based on 9 papers (2000-2009): MBE-grown Mg-doped GaN"),
            MIComment("Nf: 2e17 - 6e18 cm^-3")
            )
Tensor2<double> GaN_Mg::mob(double T) const {
    double tMob = mob_RT * (T*T*2.495E-5 -T*2.268E-2 +5.557);
    return (Tensor2<double>(tMob,tMob));
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
Tensor2<double> GaN_Mg::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

MI_PROPERTY(GaN_Mg, absp,
            MISeeClass<GaN>(MaterialInfo::absp)
            )
double GaN_Mg::absp(double wl, double T) const {
    double E = phys::h_eVc1e9/wl;
    return ( (19000.+200.*Dop()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.0001*Dop()/1e18))+(330.+30.*Dop()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.0008*Dop()/1e18)) );
}

bool GaN_Mg::isEqual(const Material &other) const {
    const GaN_Mg& o = static_cast<const GaN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && o.cond_RT == this->cond_RT;
}


Tensor2<double> GaN_Mg_bulk::thermk(double T, double t) const {
    return GaN_Mg::thermk(T, INFINITY);
}


static MaterialsDB::Register<GaN_Mg> materialDB_register_Mg;

static MaterialsDB::Register<GaN_Mg_bulk> materialDB_register_GaN_Mg_bulk;


}}       // namespace plask::materials
