#include "AlGaAsSb_Te.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaAsSb_Te::name() const { return NAME; }

MI_PARENT(AlGaAsSb_Te, AlGaAsSb)

std::string AlGaAsSb_Te::str() const { return StringBuilder("Al", Al)("Ga")("As")("Sb", Sb).dopant("Te", ND); }

AlGaAsSb_Te::AlGaAsSb_Te(const Material::Composition& Comp, DopingAmountType Type, double Val): AlGaAsSb(Comp)
{
    if (Type == CARRIERS_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        ND = Val;
        // based on: Chiu (1990) Te doping (1990) Appl. Phys. Lett. (Fig. 4), fit by L. Piskorski
        if (ND <= 1e18)
            Nf_RT = ND;
        else
        {
            double tNL = log10(ND);
            double tnL = 0.383027*tNL*tNL*tNL - 22.1278*tNL*tNL + 425.212*tNL - 2700.2222;
            Nf_RT = ( pow(10.,tnL) );
        }
    }
    double mob_RT_AlSb = 30. + (200. - 30.) / (1.+pow(ND/4e17,3.25)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    double mob_RT_GaSb = 550. + (6300. - 550.) / (1.+pow(ND/2e17,0.786)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    mob_RT = 1. / (Al/mob_RT_AlSb + Ga/mob_RT_GaSb + 9.6e-7*Al*Ga); // for small amount of arsenide
}

MI_PROPERTY(AlGaAsSb_Te, mob,
            MISource("Chiu (1990) Te doping (1990) Appl. Phys. Lett. (Fig. 4)"),
            MIComment("fit by Lukasz Piskorski")
            )
Tensor2<double> AlGaAsSb_Te::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.);
    return ( Tensor2<double>(tmob,tmob) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(AlGaAsSb_Te, Nf,
            MISource("TODO"),
            MIComment("fit by Lukasz Piskorski")
            )
double AlGaAsSb_Te::Nf(double T) const {
    double tD = 0.4506*log10(Nf_RT)-7.95;
    return ( Nf_RT*pow(T/300.,tD) );
}

double AlGaAsSb_Te::Dop() const {
    return ( ND );
}

MI_PROPERTY(AlGaAsSb_Te, cond,
            MIComment("-") // TODO
            )
Tensor2<double> AlGaAsSb_Te::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * (mob(T).c00)*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType AlGaAsSb_Te::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(AlGaAsSb_Te, nr,
            MISource("Alibert, J. Appl. Phys (1991)"),
            //MIArgumentRange(MaterialInfo::wl, 620, 2560),
            MIComment("for AlGaAsSb lattice matched to GaSb")
            )
double AlGaAsSb_Te::nr(double wl, double T, double) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double tE0 = 1.89*Ga+3.2*Al-0.36*Al*Ga;
    double tEd = 24.5*Ga+28.*Al-4.4*Al*Ga;
    double tEG = 0.725*Ga+2.338*Al-0.47*Al*Ga;
    double nR300K2 = 1. + tEd/tE0 + tEd*tE*tE/pow(tE0,3.) + tEd*pow(tE,4.)/(2.*pow(tE0,3.)*(tE0*tE0-tEG*tEG)) * log((2.*tE0*tE0-tEG*tEG-tE*tE)/(tEG*tEG-tE*tE));

    double nR300K;
    if (nR300K2>0) nR300K = sqrt(nR300K2);
    else nR300K = 1.; // TODO
    //taken from n-GaSb
    double nR = nR300K - 0.029*(Nf_RT*1e-18); // -2.9e-2 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)
    double dnRdT = Al*As*4.6e-5 + Al*Sb*1.19e-5 + Ga*As*4.5e-5 + Ga*Sb*8.2e-5;
    return ( nR + nR*dnRdT*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

MI_PROPERTY(AlGaAsSb_Te, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::wl, 1600, 4700),
            MIComment("temperature dependence - assumed: (1/abs)(dabs/dT)=1e-3"),
            MIComment("fit by Lukasz Piskorski")
            )
double AlGaAsSb_Te::absp(double wl, double T) const {
    double tAbs_RT = 1e24*exp(-wl/33.) + 1.7e-24*Nf_RT*pow(wl,1.95) + pow(20.*sqrt(Nf_RT*1e-18),1.05);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool AlGaAsSb_Te::isEqual(const Material &other) const {
    const AlGaAsSb_Te& o = static_cast<const AlGaAsSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlGaAsSb::isEqual(other);
}

static MaterialsDB::Register<AlGaAsSb_Te> materialDB_register_AlGaAsSb_Te;

}} // namespace plask::materials
