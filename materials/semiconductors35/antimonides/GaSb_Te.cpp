#include "GaSb_Te.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb_Te::name() const { return NAME; }

std::string GaSb_Te::str() const { return StringBuilder("GaSb").dopant("Te", ND); }

GaSb_Te::GaSb_Te(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        ND = Val;
        if (ND <= 1e18)
            Nf_RT = ND;
        else
        {
            double tNL = log10(ND);
            double tnL = 0.499626*tNL*tNL*tNL - 28.7231*tNL*tNL + 549.517*tNL - 3480.87;
            Nf_RT = ( pow(10.,tnL) );
        }
    }
    mob_RT = 4260e-4/(1.+pow(Nf_RT/8e17,1.25)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Te, mob,
            MISource("-"),
            MIComment("fit by Lukasz Piskorski")
            )
Tensor2<double> GaSb_Te::mob(double T) const {
    //double tmobRT = 4260./(1.+pow(Nf_RT/8e17,1.25)); // (cm^2/(V*s)) (dopasowanie do danych eksp.: Lukasz)
    double tmob = mob_RT * pow(300./T,0.8);
    return ( Tensor2<double>(tmob,tmob) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Te, Nf,
            MISource("based on 3 papers 1982 - 1996"),
            MIComment("no temperature dependence")
            )
double GaSb_Te::Nf(double T) const {
    if (ND <= 1e18)
        return ND;
    else
    {
        double tNL = log10(ND);
        double tnL = 0.499626*tNL*tNL*tNL - 28.7231*tNL*tNL + 549.517*tNL - 3480.87;
        return ( pow(10.,tnL) );
    }
}

double GaSb_Te::Dop() const {
    return ( ND );
}

MI_PROPERTY(GaSb_Te, cond,
            MIComment("-") // TODO
            )
Tensor2<double> GaSb_Te::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT;
    double tCond = tCond_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaSb_Te, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264"),
            MISource("D.E. Aspnes et al., Phys. Rev. B 27 (1983) 985-1009"),
            MISource("S. Adachi, J. Appl. Phys. 66 (1989) 6030-6040"),
            MIArgumentRange(MaterialInfo::wl, 620, 2560),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("no fitting data from 827-1798nm wavelength range"),
            MIComment("basing on fig. 5a (Adachi,1989) nR(wv) relation can be used for 620-4700nm wavelength range")
            )
double GaSb_Te::nr(double wl, double T, double) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double nR300K = 0.502*tE*tE*tE - 1.216*tE*tE + 1.339*tE + 3.419;
    double nR = nR300K - 0.034*(Nf_RT*1e-18); // -3.4e-2 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)
    return ( nR + nR*8.2e-5*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

MI_PROPERTY(GaSb_Te, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::wl, 2000, 20000),
            MIComment("no temperature dependence"),
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Te::absp(double wl, double T) const {
    double N = Nf_RT*1e-18;
    double L = wl*1e-3;
    double tFCabs = 2.42*N*pow(L,2.16-0.22*N);
    double tIVCBabs = (24.1*N+12.5)*(1.24/L-(0.094*N+0.12))+(-2.05*N-0.37);
    if (tIVCBabs>0) return ( tFCabs + tIVCBabs );
    else return ( tFCabs );
}

bool GaSb_Te::isEqual(const Material &other) const {
    const GaSb_Te& o = static_cast<const GaSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaSb::isEqual(other);
}

static MaterialsDB::Register<GaSb_Te> materialDB_register_GaSb_Te;

}} // namespace plask::materials
