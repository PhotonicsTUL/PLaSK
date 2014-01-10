#include "GaSb_Si.h"

#include <cmath>
#include <plask/material/db.h>      // MaterialsDB::Register
#include <plask/material/info.h>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb_Si::name() const { return NAME; }

std::string GaSb_Si::str() const { return StringBuilder("GaSb").dopant("Si", NA); }

GaSb_Si::GaSb_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val;
    NA = Val;
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
            MISource("assumed"), // TODO
            MIComment("no temperature dependence")
            )
double GaSb_Si::Nf(double T) const {
    return ( Nf_RT );
}

double GaSb_Si::Dop() const {
    return ( NA );
}

MI_PROPERTY(GaSb_Si, cond,
            MIComment("100% donor activation assumed") // TODO
            )
Tensor2<double> GaSb_Si::cond(double T) const {
    double tmob = 190. + (875.*pow(300./T,1.7)-190.) / (1.+pow(NA/(9e17*pow(T/300.,2.7)),0.65));
    double tCond = phys::qe * Nf_RT*1e6 * tmob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(GaSb, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264"),
            MIArgumentRange(MaterialInfo::wl, 1800, 2560),
            MIComment("fit by Lukasz Piskorski")
            )
double GaSb_Si::nr(double wl, double T, double n) const {
    double nR300K = sqrt(1.+13.05e-6*wl*wl/(1e-6*wl*wl-0.32)); // 1e-3: nm-> um
    double nR = nR300K - 0.0079*(NA*1e-18); // -7.9e-3 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)

    if (wl > 1800.)
        return ( nR + nR*8.2e-5*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
    else
        return 0.;
}

bool GaSb_Si::isEqual(const Material &other) const {
    const GaSb_Si& o = static_cast<const GaSb_Si&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaSb::isEqual(other);
}

static MaterialsDB::Register<GaSb_Si> materialDB_register_GaSb_Si;

}} // namespace plask::materials
