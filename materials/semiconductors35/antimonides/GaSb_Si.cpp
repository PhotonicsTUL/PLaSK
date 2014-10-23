#include "GaSb_Si.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb_Si::name() const { return NAME; }

std::string GaSb_Si::str() const { return StringBuilder("GaSb").dopant("Si", NA); }

GaSb_Si::GaSb_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIER_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        NA = Val;
        if ( NA < pow(10.,((1.-2.27)/(-0.0731))) )
            Nf_RT = NA;
        else
            Nf_RT = ( (-0.0731*log10(NA)+2.27) * NA );
    }
    mob_RT = 190e-4 + 685e-4 / (1.+pow(NA/9e17,0.65)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Si, mob,
            MISource("D. Martin et al., Semiconductors Science and Technology 19 (2004) 1040-1052"),
            MIComment("for all dopants") // TODO
            )
Tensor2<double> GaSb_Si::mob(double T) const {
    double tmob = 190. + (875.*pow(300./T,1.7)-190.) / (1.+pow(NA/(9e17*pow(T/300.,2.7)),0.65));
    return ( Tensor2<double>(tmob*1e-4,tmob*1e-4) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Si, Nf,
            MISource("Mirowska et al., Domieszkowanie ..."), // TODO
            MIComment("no temperature dependence")
            )
double GaSb_Si::Nf(double T) const {
    if ( NA < pow(10.,((1.-2.27)/(-0.0731))) ) return NA;
    else return ( (-0.0731*log10(NA)+2.27) * NA );
}

double GaSb_Si::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaSb_Si, cond,
            MIComment("cond(T) = cond(300K)*(300/T)^d")
            )
Tensor2<double> GaSb_Si::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT;
    double tCond = tCond_RT * pow(300./T, 1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaSb_Si, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264"),
            MISource("D.E. Aspnes et al., Phys. Rev. B 27 (1983) 985-1009"),
            MISource("S. Adachi, J. Appl. Phys. 66 (1989) 6030-6040"),
            MIArgumentRange(MaterialInfo::wl, 620, 2560),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("no fitting data from 827-1798nm wavelength range"),
            MIComment("basing on fig. 5a (Adachi,1989) nR(wv) relation can be used for 620-4700nm wavelength range")
            )
double GaSb_Si::nr(double wl, double T, double) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double nR300K = 0.502*tE*tE*tE - 1.216*tE*tE + 1.339*tE + 3.419;
    double nR = nR300K - 0.0079*(Nf_RT*1e-18); // -7.9e-3 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)
    return ( nR + nR*8.2e-5*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

MI_PROPERTY(GaSb_Si, absp,
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Si::absp(double wl, double T) const {
    double tP = Nf_RT*1e-18;
    double tLam = wl*1e-3;
    double tFCabs = (42600.*pow(tP,0.17))*pow(tLam,1.8);
    return ( tFCabs );
}

bool GaSb_Si::isEqual(const Material &other) const {
    const GaSb_Si& o = static_cast<const GaSb_Si&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaSb::isEqual(other);
}

static MaterialsDB::Register<GaSb_Si> materialDB_register_GaSb_Si;

}} // namespace plask::materials
