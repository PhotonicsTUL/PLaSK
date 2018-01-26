#include "Si3N4.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Si3N4::name() const { return NAME; }

MI_PROPERTY(Si3N4, cond,
            MISource(""),
            MIComment("TODO")
            )
Tensor2<double> Si3N4::cond(double /*T*/) const {
    throw NotImplemented("cond for Si3N4");
}

MI_PROPERTY(Si3N4, thermk,
            MISource(""),
            MIComment("TODO")
            )
Tensor2<double> Si3N4::thermk(double /*T*/, double /*h*/) const {
    throw NotImplemented("thermk for Si3N4");
}

Material::ConductivityType Si3N4::condtype() const { return Material::CONDUCTIVITY_OTHER; }

MI_PROPERTY(Si3N4, nr,
            MISource("refractiveindex.info"),
            MIArgumentRange(MaterialInfo::lam, 207, 1240)
            )
double Si3N4::nr(double lam, double /*T*/, double /*n*/) const {
    double tL2 = lam*lam*1e-6;
    return ( sqrt(1+2.8939*tL2/(tL2-0.0195077089)));
}
MI_PROPERTY(Si3N4, absp,
            MISource("S. Zhou et al., Proc. SPIE 7995 (2011) 79950T"),
            MIComment("data for SiNx"),
            MIArgumentRange(MaterialInfo::lam, 9000, 11000)
            )
double Si3N4::absp(double lam, double /*T*/) const {
    double tL = lam*1e-3;
    return ( 1.06E-4*pow(tL,7.8) );
}
bool Si3N4::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<Si3N4> materialDB_register_Si3N4;

}}       // namespace plask::materials
