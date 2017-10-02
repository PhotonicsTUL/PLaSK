#include "GaSb_Si.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb_Si::name() const { return NAME; }

std::string GaSb_Si::str() const { return StringBuilder("GaSb").dopant("Si", NA); }

GaSb_Si::GaSb_Si(DopingAmountType Type, double Val) {
    if (Type == CARRIERS_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        NA = Val;
        if ( NA < pow(10.,((1.-2.27)/(-0.0731))) )
            Nf_RT = NA;
        else
            Nf_RT = ( (-0.0731*log10(NA)+2.27) * NA );
    }
    mob_RT = 95. + (565. - 95.) / (1.+pow(NA/4e18,0.85));
}

MI_PROPERTY(GaSb_Si, mob,
            MISource("D. Martin et al., Semiconductors Science and Technology 19 (2004) 1040-1052"), // TODO
            MIComment("fit by Lukasz Piskorski")
            )
Tensor2<double> GaSb_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.2);
    return ( Tensor2<double>(tmob,tmob) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Si, Nf,
            MISource("Mirowska et al., Domieszkowanie ..."), // TODO
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Si::Nf(double T) const {
    double tD;
    if (Nf_RT <= 6.4e17) tD = 1e17*pow(Nf_RT,-1.014);
    else tD = 0.088;
    return ( Nf_RT*pow(T/300.,tD) );
}

double GaSb_Si::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaSb_Si, cond,
            MIComment("cond(T) = cond(300K)*(300/T)^d")
            )
Tensor2<double> GaSb_Si::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * (mob(T).c00)*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaSb_Si, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264"),
            MISource("D.E. Aspnes et al., Phys. Rev. B 27 (1983) 985-1009"),
            MISource("S. Adachi, J. Appl. Phys. 66 (1989) 6030-6040"),
            MIArgumentRange(MaterialInfo::lam, 620, 2560),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("no fitting data from 827-1798nm wavelength range"),
            MIComment("basing on fig. 5a (Adachi,1989) nR(wv) relation can be used for 620-4700nm wavelength range")
            )
double GaSb_Si::nr(double lam, double T, double) const {
    double tE = phys::h_eVc1e9/lam; // lam -> E
    double nR300K = 0.502*tE*tE*tE - 1.216*tE*tE + 1.339*tE + 3.419;
    double nR = nR300K - 0.0074*(Nf_RT*1e-18); // -7.4e-3 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)
    return ( nR + nR*8.2e-5*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

Material::ConductivityType GaSb_Si::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(GaSb_Si, absp,
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Si::absp(double lam, double T) const {
    double tAbs_RT = 1e24*exp(-lam/33.) + 2.02e-24*Nf_RT*pow(lam,2.) + pow(20.*sqrt(Nf_RT*1e-18),1.05);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool GaSb_Si::isEqual(const Material &other) const {
    const GaSb_Si& o = static_cast<const GaSb_Si&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaSb::isEqual(other);
}

static MaterialsDB::Register<GaSb_Si> materialDB_register_GaSb_Si;

}} // namespace plask::materials
