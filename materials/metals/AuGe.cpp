#include "AuGe.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AuGe::name() const { return NAME; }

MI_PROPERTY(AuGe, cond,
            MISource("T. Som, P. Ayyub, D. Kabiraj, N. Kulkarni, V.N. Kulkarni and D.K. Avasthi, Formation of Au0.6Ge0.4 alloy induced by Au-ion irradiation of Au/Ge bilayer, Journal of Applied Physics, vol. 84, no. 2, pp. 3861-3863, 2004."),
            MIComment("no temperature dependence")
            )
Tensor2<double> AuGe::cond(double T) const {
    double tCond = 1e8;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AuGe, thermk,
            MISource("www.thinfilm.com"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AuGe::thermk(double T, double t) const {
    double tCondT = 88.34;
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(AuGe, absp,
            MISource(""),
            MIComment("TODO")
            )
double AuGe::absp(double wl, double T) const {
    return ( 1e3 );
}

MI_PROPERTY(AuGe, nr,
            MISource(""),
            MIComment("TODO")
			)
double AuGe::nr(double wl, double T) const {
    return ( 1. );
}

static MaterialsDB::Register<AuGe> materialDB_register_AuGe;

}       // namespace plask
