#include "aSi.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string aSi::name() const { return NAME; }

MI_PROPERTY(aSi, thermk,
            MISource("D.G. Cahill et al., Physical Review B 50 (1994) 6077-6081"),
            MIComment("fit from: Lukasz Piskorski, unpublished"),
            MIArgumentRange(MaterialInfo::T, 100, 400)
            )
Tensor2<double> aSi::thermk(double T, double h) const {
    double tK = -0.73*pow(T/300.,-0.67) - 0.29*(T/300.) + 2.63;
    return ( Tensor2<double>(tK, tK) );
}

MI_PROPERTY(aSi, nr,
            MISource("R.N. Kre et al., International Journal of the Physical Sciences 5 (2010) 675-682"),
            MIArgumentRange(MaterialInfo::wl, 1050, 2050),
            MIComment("fit by Lukasz Piskorski")
            )
double aSi::nr(double wl, double T, double n) const {
    double nR300K = 0.27/pow(wl*1e-3,4.1)+3.835; // 1e-3: nm-> um

    if (wl > 1050.)
        return ( nR300K + (8.5e-5/pow(wl*1e-3,5.9))*(T-300.) ); // fit by L. Piskorski, based on "N. Do et al., Appl. Phys. Lett. 60 (1992) 2186-2188"
    else
        return 0.;
}

bool aSi::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSi> materialDB_register_aSi;

}}       // namespace plask::materials
