#include "Pt.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

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
            MIComment("TODO"),
            MIArgumentRange(MaterialInfo::wl, 400, 12400)
            )
double Pt::absp(double wl, double T) const {
    double Wl = wl*1e-3;
    if (wl<2500.)
        return ( 1e6*(2.80215 - 15.3234*Wl + 51.5342*Wl*Wl -94.3547*pow(Wl,3.) + 101.1011*pow(Wl,4.) -65.11963*pow(Wl,5.) + 24.741*pow(Wl,6.) - 5.099038*pow(Wl,7.) + 0.4391658*pow(Wl,8.)) );
    else
        return ( -39538.4 + 305946*Wl - 67838.1*Wl*Wl + 7492.84*pow(Wl,3.) - 417.401*pow(Wl,4.) + 9.27859*pow(Wl,5.) );
}

bool Pt::isEqual(const Material &other) const {
    return true;
}

MI_PROPERTY(Pt, nr,
            MISource(""),
            MIComment("TODO"),
            MIArgumentRange(MaterialInfo::wl, 280, 12400)
			)
double Pt::nr(double wl, double T, double n) const {
    double Wl = wl*1e-3;
    if (wl<3700.)
        return ( 2.20873*exp(-2.70386*pow(Wl-1.76515,2.)) + 0.438205 + 3.87609*Wl - 1.5836*Wl*Wl+0.197125*pow(Wl,3.) );
    else
        return ( 3.43266 - 0.963058*Wl + 0.260552*Wl*Wl - 0.00791393*pow(Wl,3.) );
}

static MaterialsDB::Register<Pt> materialDB_register_Pt;

}}       // namespace plask::materials
