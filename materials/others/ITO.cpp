#include "ITO.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string ITO::name() const { return NAME; }

MI_PROPERTY(ITO, cond,
            MISource("O. Tuna et al., J. Phys. D: Appl. Phys. 43 (2010) 055402 (7pp).")
            )
Tensor2<double> ITO::cond(double T) const {
    double tCond = 1e6; // (S/m)
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(ITO, thermk,
            MISource("T. Yagi et al. J. Vac. Sci. Technol. A 23 (2005) 1180-1186."),
            MIComment("for bulk ITO thermal conductivity is about 14 W/mK")
            )
Tensor2<double> ITO::thermk(double T, double t) const {
    double tCondT = 3.2;
    return ( Tensor2<double>(tCondT, tCondT) );
}

/*MI_PROPERTY(Au, absp,
            MISource(""),
            MIComment("TODO"),
            MIArgumentRange(MaterialInfo::wl, 490, 10000)
            )
double Au::absp(double wl, double T) const {
    double Wl = wl*1e-3;
    return ( -39949.7*pow(Wl,-3.07546) - 113.313*Wl*Wl - 4530.42*Wl + 816908 );
}*/

bool ITO::isEqual(const Material &other) const {
    return true;
}

/*MI_PROPERTY(Au, nr,
            MISource(""),
            MIComment("TODO"),
            MIArgumentRange(MaterialInfo::wl, 700, 10000)
			)
double Au::nr(double wl, double T, double n) const {
    double Wl = wl*1e-3;
    return ( 0.113018*pow(Wl,1.96113) + 0.185598*Wl );
}*/

static MaterialsDB::Register<ITO> materialDB_register_ITO;

}}       // namespace plask::materials
