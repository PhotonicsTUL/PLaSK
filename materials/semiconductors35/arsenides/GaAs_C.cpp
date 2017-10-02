#include "GaAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaAs_C::name() const { return NAME; }

std::string GaAs_C::str() const { return StringBuilder("GaAs").dopant("C", NA); }

GaAs_C::GaAs_C(DopingAmountType Type, double Val) {
    if (Type == CARRIERS_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.92;
    }
    else {
        Nf_RT = 0.92*Val;
        NA = Val;
    }
    mob_RT = 530./(1+pow((Nf_RT/1e17),0.30));
}

MI_PROPERTY(GaAs_C, EactA,
            MISource("R. Heilman et al., Semicond. Sci. Technol. 5 (1990) 1040-1045")
            )
double GaAs_C::EactA(double T) const {
    return 26.7e-3;
}

MI_PROPERTY(GaAs_C, EactD,
            MIComment("this parameter will be removed")
            )
double GaAs_C::EactD(double T) const {
    return 0.;
}

MI_PROPERTY(GaAs_C, mob,
            MISource("fit to p-GaAs:C (based on 23 papers 1988 - 2006)"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_C::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.25);
    return (Tensor2<double>(mob_T,mob_T));
}

MI_PROPERTY(GaAs_C, Nf,
            MISource("TODO"),
            MIComment("no temperature dependence")
            )
double GaAs_C::Nf(double T) const {
    return ( Nf_RT );
}

MI_PROPERTY(GaAs_C, Na,
            MIComment("-")
            )
double GaAs_C::Na() const {
    return ( NA );
}

MI_PROPERTY(GaAs_C, Nd,
            MIComment("-")
            )
double GaAs_C::Nd() const {
    return ( 0. );
}

double GaAs_C::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaAs_C, cond,
			MIComment("no temperature dependence")
            )
Tensor2<double> GaAs_C::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType GaAs_C::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(GaAs_C, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double GaAs_C::absp(double lam, double T) const {
    double tDWl = phys::h_eVc1e9*(Eg(300.,0.,'G')-Eg(T,0.,'G'))/(Eg(300.,0.,'G')*Eg(T,0.,'G'));
    double tWl = (lam-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0173)+0.114*pow(tWl,4.00)+73.*exp(-0.76*pow(tWl-2.74,2.)));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(0.589*pow(tWl,3.)-22.87*pow(tWl,2.)+308.*tWl-1004.14);
    return ( tAbsp );
}

bool GaAs_C::isEqual(const Material &other) const {
    const GaAs_C& o = static_cast<const GaAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaAs::isEqual(other);
}

static MaterialsDB::Register<GaAs_C> materialDB_register_GaAs_C;

}} // namespace plask::materials
