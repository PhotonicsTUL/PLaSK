#include "GaSb_Te.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb_Te::name() const { return NAME; }

std::string GaSb_Te::str() const { return StringBuilder("GaSb").dopant("Te", ND); }

GaSb_Te::GaSb_Te(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 1050e-4 + 4600e-4 / (1.+pow(ND/2.8e17,1.05)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Te, mob,
            MISource("D. Martin et al., Semiconductors Science and Technology 19 (2004) 1040-1052"),
            MIComment("for all dopants") // TODO
            )
Tensor2<double> GaSb_Te::mob(double T) const {
    double tmob = 1050. + (5650.*pow(300./T,2.0)-1050.) / (1.+pow(ND/(2.8e17*pow(T/300.,2.8)),1.05));
    return ( Tensor2<double>(tmob*1e-4,tmob*1e-4) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Te, Nf,
            MISource("based on 3 papers 1982 - 1996"),
            MIComment("no temperature dependence")
            )
double GaSb_Te::Nf(double T) const {
    return ( Nf_RT );
}

double GaSb_Te::Dop() const {
    return ( ND );
}

MI_PROPERTY(GaSb_Te, cond,
            MIComment("100% donor activation assumed") // TODO
            )
Tensor2<double> GaSb_Te::cond(double T) const {
    double tmob = 1050. + (5650.*pow(300./T,2.0)-1050.) / (1.+pow(ND/(2.8e17*pow(T/300.,2.8)),1.05));
    double tCond = phys::qe * Nf_RT*1e6 * tmob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaSb, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264"),
            MIArgumentRange(MaterialInfo::wl, 1800, 2560),
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Te::nr(double wl, double T) const {
    double nR300K = sqrt(1.+13.05e-6*wl*wl/(1e-6*wl*wl-0.32)); // 1e-3: nm-> um
    double nR = nR300K - 0.034*(ND*1e-18); // -3.4e-2 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)

    if (wl > 1800.)
        return ( nR + nR*8.2e-5*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
    else
        return 0.;
}

MI_PROPERTY(GaSb, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::wl, 2000, 20000),
            MIComment("no temperature dependence"),
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Te::absp(double wl, double T) const {
    double N = ND*1e-18;
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
