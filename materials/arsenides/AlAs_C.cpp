#include "AlAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlAs_C::name() const { return NAME; }

std::string AlAs_C::str() const { return StringBuilder("AlAs").dopant("C", NA); }

AlAs_C::AlAs_C(DopingAmountType Type, double Val) {
    //double act_GaAs = 0.92;
    //double fx1 = 1.;
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        NA = Val/0.92; // Val/(act_GaAs*fx1);
    }
    else {
        Nf_RT = 0.92*Val; // (act_GaAs*fx1)*Val;
        NA = Val;
    }
    double mob_RT_GaAs = 530e-4/(1+pow((Nf_RT/1e17),0.30)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    //double Al = 1.; // AlAs (not AlGaAs)
    double fx2 = 0.66 / (1. + pow(1./0.21,3.)) + 0.34; // (1.00-0.34) / (1. + pow(Al/0.21,3.)) + 0.34;
    mob_RT = mob_RT_GaAs * fx2;
}

MI_PROPERTY(AlAs_C, mob,
            MIComment("TODO")
            )
Tensor2<double> AlAs_C::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.25);
    return ( Tensor2<double>(mob_T,mob_T) );
}

MI_PROPERTY(AlAs_C, Nf,
            MIComment("TODO")
            )
double AlAs_C::Nf(double T) const {
    return ( Nf_RT );
}

double AlAs_C::Dop() const {
    return ( NA );
}

MI_PROPERTY(AlAs_C, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs_C::cond(double T) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AlAs_C, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double AlAs_C::absp(double wl, double T) const {
    double tEgRef300 = phys::Varshni(1.519, 0.5405e-3, 204., 300.);
    double tEgT = Eg(T,0.,'X');
    double tDWl = 1240.*(tEgRef300-tEgT)/(tEgRef300*tEgT);
    double tWl = (wl-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0173)+0.114*pow(tWl,4.00)+73.*exp(-0.76*pow(tWl-2.74,2.)));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(0.589*pow(tWl,3.)-22.87*pow(tWl,2.)+308.*tWl-1004.14);
    return ( tAbsp );
}

bool AlAs_C::isEqual(const Material &other) const {
    const AlAs_C& o = static_cast<const AlAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAs::isEqual(other);
}

static MaterialsDB::Register<AlAs_C> materialDB_register_AlAs_C;

}} // namespace plask::materials
