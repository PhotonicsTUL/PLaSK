#include "Cu.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Cu::name() const { return NAME; }

MI_PROPERTY(Cu, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: ?Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Cu::cond(double T) const {
    double tCond = 1. / (6.81e-11*(T-300.)+1.726e-8);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Cu, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Cu::thermk(double T, double t) const {
    double tCondT = 400.8*pow((300./T),0.073);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(Cu, absp,
            MISource(""),
            MIComment("TODO")
            )
double Cu::absp(double wl, double T) const {
    return ( 1e3 );
}

bool Cu::isEqual(const Material &other) const {
    return true;
}

MI_PROPERTY(Cu, nr,
            MISource(""),
            MIComment("TODO")
			)
double Cu::nr(double wl, double T, double n) const {
    return ( 1. );
}

static MaterialsDB::Register<Cu> materialDB_register_Cu;

}}       // namespace plask::materials
