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

bool GaSb_Te::isEqual(const Material &other) const {
    const GaSb_Te& o = static_cast<const GaSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaSb::isEqual(other);
}

static MaterialsDB::Register<GaSb_Te> materialDB_register_GaSb_Te;

}} // namespace plask::materials
