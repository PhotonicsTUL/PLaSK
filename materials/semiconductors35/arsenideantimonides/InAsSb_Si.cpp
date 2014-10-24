#include "InAsSb_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAsSb_Si::name() const { return NAME; }

std::string InAsSb_Si::str() const { return StringBuilder("In")("As")("Sb", Sb).dopant("Si", ND); }

MI_PARENT(InAsSb_Si, InAsSb)

InAsSb_Si::InAsSb_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): InAsSb(Comp)//, mGaAs_Si(Type,Val), mAlAs_Si(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        ND = Val;
        if (ND <= 1e18) // assumed (no data)
            Nf_RT = ND;
        else
            Nf_RT = (0.63*(ND*1e-18-1)+1)*1e18;
    }
    double tInSb_mob_RT = 60000e-4/(1.+pow((Nf_RT/8e16),0.73));
    double tInAs_mob_RT = 15000e-4/(1.+pow((Nf_RT/1e18),0.81));
    mob_RT = 1./(As/tInAs_mob_RT+Sb/tInSb_mob_RT) + As*Sb*7000.;
}

MI_PROPERTY(InAsSb_Si, mob,
            MISource("T. Taliercio, Optics Express 22 (2014) pp. 24294-24303"),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("fitted for high doping and high As content"),
            MIComment("mob(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAsSb_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tmob, tmob) );
}

MI_PROPERTY(InAsSb_Si, Nf,
            MISource("T. Taliercio, Optics Express 22 (2014) pp. 24294-24303"),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("no temperature dependence")
            )
double InAsSb_Si::Nf(double T) const {
    return ( Nf_RT );
}

double InAsSb_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InAsSb_Si, cond,
            MIComment("cond(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAsSb_Si::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT;
    double tCond = tCond_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(InAsSb_Si, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211"),
            MIArgumentRange(MaterialInfo::wl, 500, 7000),
            MIComment("TODO")
            )
double InAsSb_Si::nr(double wl, double T, double n) const {
    double nR300K = sqrt(1.+8.75e-6*wl*wl/(1e-6*wl*wl-0.15)); // 1e-3: nm-> um
    double nR = nR300K - 0.034*(Nf_RT*1e-18); // -3.4e-2 - the same as for GaSb TODO

    if (wl > 500.)
        return ( nR + nR*(As*4.6e-5+Sb*1.19e-5)*(T-300.) ); // 4.6e-5, 1.19e-5 - from Adachi (2005) ebook p.243 tab. 10.6
    else
        return 0.;
}

MI_PROPERTY(InAsSb, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::wl, 2000, 20000),
            MIComment("no temperature dependence"),
            MIComment("taken from n-GaSb:Te")
            )
double InAsSb_Si::absp(double wl, double T) const {
    double N = Nf_RT*1e-18;
    double L = wl*1e-3;
    double tFCabs = 2.42*N*pow(L,2.16-0.22*N);
    double tIVCBabs = (24.1*N+12.5)*(1.24/L-(0.094*N+0.12))+(-2.05*N-0.37);
    if (tIVCBabs>0) return ( tFCabs + tIVCBabs );
    else return ( tFCabs );
}

bool InAsSb_Si::isEqual(const Material &other) const {
    const InAsSb_Si& o = static_cast<const InAsSb_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAsSb::isEqual(other);
}

static MaterialsDB::Register<InAsSb_Si> materialDB_register_InAsSb_Si;

}}       // namespace plask::materials
