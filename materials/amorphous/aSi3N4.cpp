#include "aSi3N4.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string aSi3N4::name() const { return NAME; }

MI_PROPERTY(aSi3N4, cond,
            MISource(""),
            MIComment("TODO")
            )
Tensor2<double> aSi3N4::cond(double T) const {
    return ( Tensor2<double>(0., 0.) ); //TODO
}

MI_PROPERTY(aSi3N4, thermk,
            MISource(""),
            MIComment("TODO")
            )
Tensor2<double> aSi3N4::thermk(double T, double h) const {
    return ( Tensor2<double>(0., 0.) ); //TODO
}

MI_PROPERTY(aSi3N4, nr,
            MISource("refractiveindex.info"),
            MIArgumentRange(MaterialInfo::wl, 207, 1240)
            )
double aSi3N4::nr(double wl, double T, double n) const {
    double tL2 = wl*wl*1e-6;
    return ( sqrt(1+2.8939*tL2/(tL2-0.0195077089)));
}
MI_PROPERTY(aSi3N4, absp,
            MISource("S. Zhou et al., Proc. SPIE 7995 (2011) 79950T"),
            MIComment("data for SiNx"),
            MIArgumentRange(MaterialInfo::wl, 9000, 11000)
            )
double aSi3N4::absp(double wl, double T) const {
    double tL = wl*1e-3;
    return ( 1.06E-4*pow(tL,7.8) );
}
bool aSi3N4::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSi3N4> materialDB_register_aSi3N4;

}}       // namespace plask::materials
