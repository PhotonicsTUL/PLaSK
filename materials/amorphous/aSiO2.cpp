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
            MIComment("fit from: Lukasz Piskorski, unpublished"),
            MIArgumentRange(MaterialInfo::T, 77, 750)
            )
Tensor2<double> aSiO2::thermk(double T, double h) const {
    double tK;
    if (T < 750.) tK = 0.303*pow(T/300.,0.0194) - 1.9e-4*pow(T/300.,2.) - 0.2899; // [tK] = W/(cm*K)
    else tK = 4.81150e-6 * T + 0.013738175; // [tK] = W/(cm*K)
    return ( Tensor2<double>(tK*100., tK*100.) );
}

Material::ConductivityType aSiO2::condtype() const { return Material::CONDUCTIVITY_OTHER; }

MI_PROPERTY(aSiO2, nr,
            MISource("I.H. Malitson, Journal of the Optical Society of America 55 (1965) 1205-1209"),
            MIArgumentRange(MaterialInfo::wl, 210, 3710)
            )
double aSiO2::nr(double wl, double T, double n) const {
    double L = wl*1e-3;
    double nR293K = sqrt( 1. + 0.6961663*L*L/(L*L-pow(0.0684043,2.)) + 0.4079426*L*L/(L*L-pow(0.1162414,2.)) + 0.8974794*L*L/(L*L-pow(9.896161,2.)) ); // 1e-3: nm-> um
    return ( nR293K + 1.1e-5*(T-293.) ); // based on fig.3 in "I.H. Malitson, Journal of the Optical Society of America 55 (1965) 1205-1209"
}

MI_PROPERTY(aSiO2, absp,
            MISource("TODO"),
            MIArgumentRange(MaterialInfo::wl, 400, 4500),
            MIComment("temperature dependence - assumed: (1/abs)(dabs/dT)=1e-3"),
            MIComment("fit by Lukasz Piskorski")
            )
double aSiO2::absp(double wl, double T) const {
    double tAbsRTL;
    if (wl < 1173.15) tAbsRTL = -0.257 * pow(wl*1e-3,6.) - 1.72;
    else tAbsRTL = 0.982 * (wl*1e-3) - 3.542;
    double tAbsRT = pow(10.,tAbsRTL);
    return ( tAbsRT + tAbsRT*1e-3*(T-300.) );
}

MI_PROPERTY(aSiO2, eps,
            MISource("J. Robertson, Eur. Phys. J. Appl. Phys. 28, (2004) 265-291")
            )
double aSiO2::eps(double T) const {
    return 3.9;
}

bool aSiO2::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSiO2> materialDB_register_aSiO2;

}}       // namespace plask::materials
