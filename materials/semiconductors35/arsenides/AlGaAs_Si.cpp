#include "AlGaAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaAs_Si::name() const { return NAME; }

std::string AlGaAs_Si::str() const { return StringBuilder("Al", Al)("Ga")("As").dopant("Si", ND); }

MI_PARENT(AlGaAs_Si, AlGaAs)

AlGaAs_Si::AlGaAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaAs(Comp), mGaAs_Si(Type,Val), mAlAs_Si(Type,Val)
{
    double fx1A = (1.-7.8*Al*Al); // x < 0.35
    double fx1B = (1.14*Al-0.36); // else
    if (Type == CARRIERS_CONCENTRATION) {
        Nf_RT = Val;
        if (Al < 0.35) ND = mGaAs_Si.Dop()*fx1A;
        else ND = mGaAs_Si.Dop()*fx1B;
    }
    else {
        ND = Val;
        double Nf_GaAs_Si_RT = ND; // = 1.00*ND
        if (Al < 0.35) Nf_RT = Nf_GaAs_Si_RT*fx1A;
        else Nf_RT = Nf_GaAs_Si_RT*fx1B;
    }
    double mob_GaAs_Si_RT = 6600./(1+pow((Nf_RT/5e17),0.53));
    double fx2A = exp(-16.*Al*Al); // x < 0.5
    double fx2B = 0.054*Al-0.009; // else
    if (Al < 0.5) mob_RT = mob_GaAs_Si_RT * fx2A;
    else mob_RT = mob_GaAs_Si_RT * fx2B;
}

MI_PROPERTY(AlGaAs_Si, EactA,
            MIComment("this parameter will be removed")
            )
double AlGaAs_Si::EactA(double T) const {
    return 0.;
}

MI_PROPERTY(AlGaAs_Si, EactD,
            MISource("L. Piskorski, PhD thesis (2010)")
            )
double AlGaAs_Si::EactD(double T) const {
    return 1e-3; // TODO add correct value
}

MI_PROPERTY(AlGaAs_Si, mob,
            MISource("based on 3 papers 1982-1990 about Si-doped AlGaAs"),
            MISource("based on Si-doped GaAs")
            )
Tensor2<double> AlGaAs_Si::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.4);
    return ( Tensor2<double>(mob_T, mob_T) );
}

MI_PROPERTY(AlGaAs_Si, Nf,
            MISource("based on 2 papers 1982, 1984 about Si-doped AlGaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs_Si::Nf(double T) const {
    return ( Nf_RT );
}

MI_PROPERTY(AlGaAs_Si, Na,
            MIComment("-")
            )
double AlGaAs_Si::Na() const {
    return ( 0. );
}

MI_PROPERTY(AlGaAs_Si, Nd,
            MIComment("-")
            )
double AlGaAs_Si::Nd() const {
    return ( ND );
}

double AlGaAs_Si::Dop() const {
    return ( ND );
}

Tensor2<double> AlGaAs_Si::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType AlGaAs_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(AlGaAs_Si, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double AlGaAs_Si::absp(double lam, double T) const {
    double tEgRef300 = mGaAs_Si.Eg(300.,0.,'G');
    double tEgT = Eg(T,0.,'G');
    if (tEgT > Eg(T,0.,'X'))
        tEgT = Eg(T,0.,'X');
    double tDWl = phys::h_eVc1e9*(tEgRef300-tEgT)/(tEgRef300*tEgT);
    double tWl = (lam-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0169)+4.67+0.00211*pow(tWl,4.80));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(-8.4+0.233*pow(tWl,2.6));
    return ( tAbsp );
}

bool AlGaAs_Si::isEqual(const Material &other) const {
    const AlGaAs_Si& o = static_cast<const AlGaAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlGaAs::isEqual(other);
}

static MaterialsDB::Register<AlGaAs_Si> materialDB_register_AlGaAs_Si;

}}       // namespace plask::materials
