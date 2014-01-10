#include "aSiO2.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string aSiO2::name() const { return NAME; }

MI_PROPERTY(aSiO2, cond,
            MISource("www.siliconfareast.com/sio2si3n4.htm"),
            MIComment("no temperature dependence")
            )
Tensor2<double> aSiO2::cond(double T) const {
    return ( Tensor2<double>(1e-13, 1e-13) );
}

MI_PROPERTY(aSiO2, thermk,
            MISource("D.G. Cahill et al., Review of Scientific Instruments 61 (1990) 802-808"),
            MIComment("fit from: Lukasz Piskorski, unpublished"),
            MIArgumentRange(MaterialInfo::T, 200, 670)
            )
Tensor2<double> aSiO2::thermk(double T, double h) const {
    double tK = 2.66*pow(T/300.,0.53) - 0.82*(T/300.) - 0.55;
    return ( Tensor2<double>(tK, tK) );
}

MI_PROPERTY(aSiO2, nr,
            MISource("I.H. Malitson, Journal of the Optical Society of America 55 (1965) 1205-1209"),
            MIArgumentRange(MaterialInfo::wl, 210, 3710)
            )
double aSiO2::nr(double wl, double T, double n) const {
    double L = wl*1e-3;
    double nR293K = sqrt(1.+0.6961663*L*L/(L*L-pow(0.0684043,2.))
                         +0.4079426*L*L/(L*L-pow(0.1162414,2.))
                         +0.8974794*L*L/(L*L-pow(9.896161,2.))); // 1e-3: nm-> um

    if (wl > 210.)
        return ( nR293K + ((0.08*L+0.01)/pow(L,3.5)+13.2*pow(L,0.69)-7.4*L+4.9)*1e-6*(T-293.) ); // fit by L. Piskorski, based on "I.H. Malitson, Journal of the Optical Society of America 55 (1965) 1205-1209"
    else
        return 0.;
}

bool aSiO2::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSiO2> materialDB_register_aSiO2;

}}       // namespace plask::materials
