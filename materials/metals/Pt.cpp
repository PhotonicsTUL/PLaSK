#include "Pt.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string Pt::name() const { return NAME; }

MI_PROPERTY(Pt, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: ?Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Pt::cond(double T) const {
    double tCond = 1. / (3.84e-10*(T-300.)+1.071e-7);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Pt, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Pt::thermk(double T, double t) const {
    double tCondT = 3.6e-5*pow(T-300.,2.) - 4e-3*(T-300.) + 71.7;
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(Pt, absp,
            MISource(""),
            MIComment("TODO")
            )
double Pt::absp(double wl, double T) const {
    return ( 1e3 );
}

MI_PROPERTY(Pt, nr,
            MISource(""),
            MIComment("TODO")
			)
double Pt::nr(double wl, double T) const {
    return ( 1. );
}

static MaterialsDB::Register<Pt> materialDB_register_Pt;

}       // namespace plask
