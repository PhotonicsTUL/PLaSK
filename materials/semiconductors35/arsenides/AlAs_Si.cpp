#include "AlAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlAs_Si::name() const { return NAME; }

std::string AlAs_Si::str() const { return StringBuilder("AlAs").dopant("Si", ND); }

AlAs_Si::AlAs_Si(DopingAmountType Type, double Val) {
    //double act_GaAs = 1.;
    //double Al = 1.; // AlAs (not AlGaAs)
    //double fx1 = 1.14*Al-0.36;
    if (Type == CARRIER_CONCENTRATION) {
        Nf_RT = Val;
        ND = Val/0.78; // Val/(act_GaAs*fx1);
    }
    else {
        Nf_RT = 0.78*Val; // (act_GaAs*fx1)*Val;
        ND = Val;
    }
    double mob_RT_GaAs = 6600./(1+pow((Nf_RT/5e17),0.53));
    double fx2 = 0.045; // 0.054*Al-0.009;
    mob_RT = mob_RT_GaAs * fx2;
}

MI_PROPERTY(AlAs_Si, mob,
            MIComment("TODO")
            )
Tensor2<double> AlAs_Si::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.4);
    return ( Tensor2<double>(mob_T,mob_T) );
}

MI_PROPERTY(AlAs_Si, Nf,
            MIComment("TODO")
            )
double AlAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

double AlAs_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(AlAs_Si, cond,
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAs_Si::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType AlAs_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(AlAs_Si, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double AlAs_Si::absp(double wl, double T) const {
    double tEgRef300 = phys::Varshni(1.519, 0.5405e-3, 204., T);
    double tEgT = Eg(T,0.,'X');
    double tDWl = phys::h_eVc1e9*(tEgRef300-tEgT)/(tEgRef300*tEgT);
    double tWl = (wl-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0169)+4.67+0.00211*pow(tWl,4.80));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(-8.4+0.233*pow(tWl,2.6));
    return ( tAbsp );
}

bool AlAs_Si::isEqual(const Material &other) const {
    const AlAs_Si& o = static_cast<const AlAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAs::isEqual(other);
}

static MaterialsDB::Register<AlAs_Si> materialDB_register_AlAs_Si;

}} // namespace plask::materials
