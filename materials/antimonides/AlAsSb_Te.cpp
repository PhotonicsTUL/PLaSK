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
    if (Type == CARRIER_CONCENTRATION) Nf_RT = Val;
    else ND = Val;
    mob_RT = 2000.;
}

MI_PROPERTY(AlAsSb_Te, mob,
            MISource("-"),
            MISource("TODO")
            )
Tensor2<double> AlAsSb_Te::mob(double T) const {
    return ( Tensor2<double>(mob_RT, mob_RT) ); // TODO
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
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AlAsSb_Te, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211"),
            MIArgumentRange(MaterialInfo::wl, 500, 7000),
            MIComment("TODO")
            )
double AlAsSb_Te::nr(double wl, double T) const {
    double nR300K = sqrt(1.+8.75e-6*wl*wl/(1e-6*wl*wl-0.15)); // 1e-3: nm-> um
    double nR = nR300K - 0.034*(ND*1e-18); // -3.4e-2 - the same as for GaSb TODO

    if (wl > 500.)
        return ( nR + nR*(As*4.6e-5+Sb*1.19e-5)*(T-300.) ); // 4.6e-5, 1.19e-5 - from Adachi (2005) ebook p.243 tab. 10.6
    else
        return 0.;
}

bool AlAsSb_Te::isEqual(const Material &other) const {
    const AlAsSb_Te& o = static_cast<const AlAsSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAsSb::isEqual(other);
}

static MaterialsDB::Register<AlAsSb_Te> materialDB_register_AlAsSb_Te;

}}       // namespace plask::materials
