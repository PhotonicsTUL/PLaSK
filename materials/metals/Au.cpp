#include "Au.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Au::name() const { return NAME; }

MI_PROPERTY(Au, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Au::cond(double T) const {
    double tCond = 1. / (8.38e-11*(T-300.)+2.279e-8);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Au, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Au::thermk(double T, double /*t*/) const {
    double tCondT = -0.064*(T-300.)+317.1;
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(Au, absp,
            MISource(""),
            MIComment("TODO"),
            MIArgumentRange(MaterialInfo::lam, 490, 10000)
            )
double Au::absp(double lam, double /*T*/) const {
    double ulam = lam*1e-3;
    return ( -39949.7*pow(ulam,-3.07546) - 113.313*ulam*ulam - 4530.42*ulam + 816908 );
}

bool Au::isEqual(const Material &/*other*/) const {
    return true;
}

MI_PROPERTY(Au, nr,
            MISource(""),
            MIComment("TODO"),
            MIArgumentRange(MaterialInfo::lam, 700, 10000)
			)
double Au::nr(double lam, double /*T*/, double /*n*/) const {
    double ulam = lam*1e-3;
    return ( 0.113018*pow(ulam,1.96113) + 0.185598*ulam );
}

static MaterialsDB::Register<Au> materialDB_register_Au;

}}       // namespace plask::materials
