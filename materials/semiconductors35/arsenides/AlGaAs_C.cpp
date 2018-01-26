#include "AlGaAs_C.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaAs_C::name() const { return NAME; }

std::string AlGaAs_C::str() const { return StringBuilder("Al", Al)("Ga")("As").dopant("C", NA); }

MI_PARENT(AlGaAs_C, AlGaAs)

AlGaAs_C::AlGaAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaAs(Comp), mGaAs_C(Type,Val), mAlAs_C(Type,Val)
{
    if (Type == CARRIERS_CONCENTRATION) {
        Nf_RT = Val;
        NA = mGaAs_C.Dop(); // mGaAs_C.Dop()*fx1;
    }
    else {
        NA = Val;
        //double Nf_GaAs_C_RT = 0.92*NA; do not delete this!
        Nf_RT = 0.92*NA/*Nf_GaAs_C_RT*/; // fx1 = 1
    }
    double mob_GaAs_C_RT = 530./(1+pow((Nf_RT/1e17),0.30));
    double fx = 0.66 / (1. + pow(Al/0.21,3.0)) + 0.34; // (1.00-0.34) / (1. + pow(Al/0.21,3.0)) + 0.34;
    mob_RT = mob_GaAs_C_RT * fx;
}

MI_PROPERTY(AlGaAs_C, EactA,
            MISource("R. Heilman et al., Semicond. Sci. Technol. 5 (1990) 1040-1045")
            )
double AlGaAs_C::EactA(double /*T*/) const {
    return 26.7e-3; // TODO add correct value
}

MI_PROPERTY(AlGaAs_C, EactD,
            MIComment("this parameter will be removed")
            )
double AlGaAs_C::EactD(double /*T*/) const {
    return 0.;
}

MI_PROPERTY(AlGaAs_C, mob,
            MISource("based on 4 papers 1991-2000 about C-doped AlGaAs"),
            MISource("based on C-doped GaAs")
            )
Tensor2<double> AlGaAs_C::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.25);
    return ( Tensor2<double>(mob_T, mob_T) );
}

MI_PROPERTY(AlGaAs_C, Nf,
            MISource("based on 3 papers 1991-2004 about C-doped AlGaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs_C::Nf(double /*T*/) const {
    return ( Nf_RT );
}

MI_PROPERTY(AlGaAs_C, Na,
            MIComment("-")
            )
double AlGaAs_C::Na() const {
    return ( NA );
}

MI_PROPERTY(AlGaAs_C, Nd,
            MIComment("-")
            )
double AlGaAs_C::Nd() const {
    return ( 0. );
}

double AlGaAs_C::Dop() const {
    return ( NA );
}

Tensor2<double> AlGaAs_C::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType AlGaAs_C::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(AlGaAs_C, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double AlGaAs_C::absp(double lam, double T) const {
    double tEgRef300 = mGaAs_C.Eg(300.,0.,'G');
    double tEgT = Eg(T,0.,'G');
    if (tEgT > Eg(T,0.,'X'))
        tEgT = Eg(T,0.,'X');
    double tDWl = phys::h_eVc1e9*(tEgRef300-tEgT)/(tEgRef300*tEgT);
    double tWl = (lam-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0173)+0.114*pow(tWl,4.00)+73.*exp(-0.76*pow(tWl-2.74,2.)));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(0.589*pow(tWl,3.)-22.87*pow(tWl,2.)+308.*tWl-1004.14);
    return ( tAbsp );
}

bool AlGaAs_C::isEqual(const Material &other) const {
    const AlGaAs_C& o = static_cast<const AlGaAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlGaAs::isEqual(other);
}

static MaterialsDB::Register<AlGaAs_C> materialDB_register_AlGaAs_C;

}} // namespace plask::materials
