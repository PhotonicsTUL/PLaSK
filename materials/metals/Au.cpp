#include "Au.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string Au::name() const { return NAME; }

MI_PROPERTY(Au, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
std::pair<double,double> Au::cond(double T) const {
    double tCond = 1. / (8.38e-11*(T-300.)+2.279e-8);
    return ( std::make_pair(tCond, tCond) );
}

MI_PROPERTY(Au, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
std::pair<double,double> Au::thermk(double T, double t) const {
    double tCondT = -0.064*(T-300.)+317.1;
    return ( std::make_pair(tCondT, tCondT) );
}

MI_PROPERTY(Au, absp,
            MISource(""),
            MIComment("TODO")
            )
double Au::absp(double wl, double T) const {
    return ( 1e3 );
}

MI_PROPERTY(Au, nr,
            MISource(""),
            MIComment("TODO")
			)
double Au::nr(double wl, double T) const {
    return ( 1. );
}

static MaterialsDB::Register<Au> materialDB_register_Au;

}       // namespace plask
