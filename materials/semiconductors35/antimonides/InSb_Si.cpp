#include "InSb_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InSb_Si::name() const { return NAME; }

std::string InSb_Si::str() const { return StringBuilder("InSb").dopant("Si", ND); }

InSb_Si::InSb_Si(DopingAmountType Type, double Val) {
    Nf_RT = Val; //TODO
    ND = Val; //TODO
    mob_RT = 60000./(1.+pow((Nf_RT/8e16),0.73));
}

MI_PROPERTY(InSb_Si, mob,
            MISource("R.S. Popovic, Hall Effect Devices, CRC Press, 2003 (fig. 2.10, p. 45)"),
            MISource("J.E. Oh et al., J. Appl. Phys. 66 (1989) 3618-3621"),
            MISource("M. Henini, Molecular Beam Epitaxy: From Research to Mass Production, Newnes, 2012 (fig. 31.26, p. 712)"),
            MIComment("mob(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InSb_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,0.8);
    return ( Tensor2<double>(tmob,tmob) );
}

MI_PROPERTY(InSb_Si, Nf,
            MIComment("Nf(ND) assumed, TODO: find exp. data"),
            MIComment("no temperature dependence")
            )
double InSb_Si::Nf(double T) const {
    return ( Nf_RT );
}

double InSb_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InSb_Si, cond,
            MIComment("cond(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InSb_Si::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT*1e-4;
    double tCond = tCond_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType InSb_Si::condtype() const { return Material::CONDUCTIVITY_N; }

bool InSb_Si::isEqual(const Material &other) const {
    const InSb_Si& o = static_cast<const InSb_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InSb::isEqual(other);
}

static MaterialsDB::Register<InSb_Si> materialDB_register_InSb_Si;

}} // namespace plask::materials
