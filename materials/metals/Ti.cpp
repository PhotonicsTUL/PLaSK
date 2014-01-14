#include "Ti.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Ti::name() const { return NAME; }

MI_PROPERTY(Ti, cond,
            MISource("N.D. Milosevic and K.D. Maglic, Thermophysical properties of solid phase titanium in a wide temperature range, High Temperatures-High Pressures, vol. 37, pp. 187-204, 2008."),
            MIArgumentRange(MaterialInfo::T, 250, 1100)
            )
Tensor2<double> Ti::cond(double T) const {
    double tCond = 1. / (6.17169e-8 + 9.010579e-10*T + 1.817669e-12*T*T - 1.225226e-15*T*T*T);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Ti, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Ti::thermk(double T, double t) const {
    double tCondT = 22.00*pow((300./T),0.24);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(Ti, absp,
            MISource(""),
            MIComment("TODO")
            )
double Ti::absp(double wl, double T) const {
    return ( 1e3 );
}

bool Ti::isEqual(const Material &other) const {
    return true;
}

MI_PROPERTY(Ti, nr,
            MISource(""),
            MIComment("TODO")
			)
double Ti::nr(double wl, double T, double n) const {
    return ( 1. );
}

static MaterialsDB::Register<Ti> materialDB_register_Ti;

}}       // namespace plask::materials
