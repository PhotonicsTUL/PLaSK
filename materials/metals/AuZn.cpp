#include "AuZn.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AuZn::name() const { return NAME; }

MI_PROPERTY(AuZn, cond,
            MISource("C. Belouet, C. Villard, C. Fages and D. Keller, Achievement of homogeneous AuSn solder by pulsed laser-assisted deposition, Journal of Electronic Materials, vol. 28, no. 10, pp. 1123-1126, 1999."),
            MIComment("no temperature dependence")
            )
Tensor2<double> AuZn::cond(double /*T*/) const {
    double tCond = 1e6; // TODO (check this value: AuZn or AuSn)
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AuZn, thermk,
            MISource("D. Singh and D.K. Pandey, Ultrasonic investigations in intermetallics, Pramana - Journal of Physics, vol. 72, no. 2, pp. 389-398, 2009."),
            MIComment("no temperature dependence")
            )
Tensor2<double> AuZn::thermk(double /*T*/, double /*t*/) const {
    double tCondT = 110.3;
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(AuZn, absp,
            MISource(""),
            MIComment("TODO")
            )
double AuZn::absp(double /*lam*/, double /*T*/) const {
    return ( 1e3 );
}

bool AuZn::isEqual(const Material &/*other*/) const {
    return true;
}

MI_PROPERTY(AuZn, nr,
            MISource(""),
            MIComment("TODO")
			)
double AuZn::nr(double /*lam*/, double /*T*/, double /*n*/) const {
    return ( 1. );
}

static MaterialsDB::Register<AuZn> materialDB_register_AuZn;

}}       // namespace plask::materials
