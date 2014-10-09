#include "AlAsSb_Te.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlAsSb_Te::name() const { return NAME; }

std::string AlAsSb_Te::str() const { return StringBuilder("Al")("As")("Sb", Sb).dopant("Si", ND); }

MI_PARENT(AlAsSb_Te, AlAsSb)

AlAsSb_Te::AlAsSb_Te(const Material::Composition& Comp, DopingAmountType Type, double Val): AlAsSb(Comp)//, mGaAs_Si(Type,Val), mAlAs_Si(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        ND = Val;
        if (ND <= 1e18) // taken from n-GaSb:Te
            Nf_RT = ND;
        else // taken from n-GaSb:Te
        {
            double tNL = log10(ND);
            double tnL = 0.499626*tNL*tNL*tNL - 28.7231*tNL*tNL + 549.517*tNL - 3480.87;
            Nf_RT = ( pow(10.,tnL) );
        }
    }
    double tAlSb_mob_RT = 240e-4/(1.+pow(Nf_RT/2e17,1.14)); // (m^2/(V*s)) (fitted by Lukasz)
    mob_RT = tAlSb_mob_RT; // Springer AlSb
}

MI_PROPERTY(AlAsSb_Te, mob,
            MISource("Springer - AlSb")
            )
Tensor2<double> AlAsSb_Te::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tmob, tmob) );
}

MI_PROPERTY(AlAsSb_Te, Nf,
            MISource("-"),
            MIComment("TODO")
            )
double AlAsSb_Te::Nf(double T) const {
    return ( Nf_RT );
}

double AlAsSb_Te::Dop() const {
    return ( ND );
}

Tensor2<double> AlAsSb_Te::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT;
    double tCond = tCond_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AlAsSb_Te, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211"),
            MIArgumentRange(MaterialInfo::wl, 500, 7000),
            MIComment("TODO")
            )
double AlAsSb_Te::nr(double wl, double T, double n) const {
    double nR300K = sqrt(1.+8.75e-6*wl*wl/(1e-6*wl*wl-0.15)); // 1e-3: nm-> um
    double nR = nR300K - 0.034*(Nf_RT*1e-18); // -3.4e-2 - the same as for GaSb TODO

    if (wl > 500.)
        return ( nR + nR*(As*4.6e-5+Sb*1.19e-5)*(T-300.) ); // 4.6e-5, 1.19e-5 - from Adachi (2005) ebook p.243 tab. 10.6
    else
        return 0.;
}

MI_PROPERTY(AlAsSb, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::wl, 2000, 20000),
            MIComment("no temperature dependence"),
            MIComment("taken from n-GaSb:Te")
            )
double AlAsSb_Te::absp(double wl, double T) const {
    double N = Nf_RT*1e-18;
    double L = wl*1e-3;
    double tFCabs = 2.42*N*pow(L,2.16-0.22*N);
    double tIVCBabs = (24.1*N+12.5)*(1.24/L-(0.094*N+0.12))+(-2.05*N-0.37);
    if (tIVCBabs>0) return ( tFCabs + tIVCBabs );
    else return ( tFCabs );
}

bool AlAsSb_Te::isEqual(const Material &other) const {
    const AlAsSb_Te& o = static_cast<const AlAsSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAsSb::isEqual(other);
}

static MaterialsDB::Register<AlAsSb_Te> materialDB_register_AlAsSb_Te;

}}       // namespace plask::materials
