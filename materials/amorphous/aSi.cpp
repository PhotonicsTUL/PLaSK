#include "aSi.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string aSi::name() const { return NAME; }

MI_PROPERTY(aSi, cond,
            MISource("A.S. Diamond, Handbook of Imaging Materials, p. 630"),
            MIComment("no temperature dependence")
            )
Tensor2<double> aSi::cond(double T) const {
    return ( Tensor2<double>(0.1, 0.1) );
}

MI_PROPERTY(aSi, thermk,
            MISource("D.G. Cahill et al., Physical Review B 50 (1994) 6077-6081"),
            MIComment("fit from: Lukasz Piskorski, unpublished"),
            MIArgumentRange(MaterialInfo::T, 77, 400)
            )
Tensor2<double> aSi::thermk(double T, double h) const {
    double tK;
    if (T<=400.) tK = 20.48*pow(T/300.,0.824) + 1.1*pow(T/300.,2.) - 18.9*(T/300.) - 1.07;
    else tK = 2.52148e-4*T + 1.5431908;
    return ( Tensor2<double>(tK, tK) );
}

Material::ConductivityType aSi::condtype() const { return Material::CONDUCTIVITY_OTHER; }

MI_PROPERTY(aSi, nr,
            MISource("R.N. Kre et al., International Journal of the Physical Sciences 5 (2010) 675-682"),
            MIArgumentRange(MaterialInfo::lam, 2200, 3900),
            MIComment("fit by Lukasz Piskorski")
            )
double aSi::nr(double lam, double T, double n) const {
    double tE = phys::h_eVc1e9/lam; // lam -> E
    double nR300K = 0.144+3.308/(1.-0.0437*tE*tE); // 1e-3: nm-> um
    return ( nR300K + 2.2e-5*(T-300.) ); // assumed (see: dnRdT for SiO2)
}

MI_PROPERTY(aSi, absp,
            MISource("TODO"),
            //MIArgumentRange(MaterialInfo::lam, 1600, 4700),
            MIComment("temperature dependence - assumed: (1/abs)(dabs/dT)=1e-3"),
            MIComment("fit by Lukasz Piskorski")
            )
double aSi::absp(double lam, double T) const {
    double tE = phys::h_eVc1e9/lam; // lam -> E
    double tAbs_RT = pow(10.,2.506*tE+0.2);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool aSi::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSi> materialDB_register_aSi;

}}       // namespace plask::materials
