#include "ITO.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string ITO::name() const { return NAME; }

MI_PROPERTY(ITO, cond,
            MISource("O. Tuna et al., J. Phys. D: Appl. Phys. 43 (2010) 055402 (7pp).")
            )
Tensor2<double> ITO::cond(double /*T*/) const {
    const double tCond = 1e6; // (S/m)
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(ITO, thermk,
            MISource("T. Yagi et al. J. Vac. Sci. Technol. A 23 (2005) 1180-1186."),
            MIComment("for bulk ITO thermal conductivity is about 14 W/mK")
            )
Tensor2<double> ITO::thermk(double /*T*/, double /*t*/) const {
    const double tCondT = 3.2;
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(ITO, absp,
            MISource("E.F. Schubert (2004) Refractive index and extinction coefficient of materials"),
            MISource("https://www.ecse.rpi.edu/~schubert/Educational-resources/Materials-Refractive-index-and-extinction-coefficient.pdf"),
            MIComment("fit: Lukasz Piskorski"),
            MIArgumentRange(MaterialInfo::lam, 340, 1400)
            )
double ITO::absp(double lam, double /*T*/) const {
    const double a6 = 2.22043e-18;
    const double a5 = -1.09082e-14;
    const double a4 = 2.21842e-11;
    const double a3 = -2.36846e-8;
    const double a2 = 1.40754e-5;
    const double a1 = -4.40237e-3;
    const double a0 = 5.75812e-1;

    const double k = a6*pow(lam,6) + a5*pow(lam,5) + a4*pow(lam,4) + a3*pow(lam,3) + a2*lam*lam + a1*lam + a0; // lam in nm

    return (4. * M_PI * k / (lam*1e-7)); // result in 1/cm
}

bool ITO::isEqual(const Material &/*other*/) const {
    return true;
}

MI_PROPERTY(ITO, nr,
            MISource("E.F. Schubert (2004) Refractive index and extinction coefficient of materials"),
            MISource("https://www.ecse.rpi.edu/~schubert/Educational-resources/Materials-Refractive-index-and-extinction-coefficient.pdf"),
            MIComment("fit: Lukasz Piskorski"),
            MIArgumentRange(MaterialInfo::lam, 340, 1400)
			)
double ITO::nr(double lam, double /*T*/, double /*n*/) const {
    const double a6 = 4.75702e-18;
    const double a5 = -2.752990e-14;
    const double a4 = 6.45504e-11;
    const double a3 = -7.91161e-8;
    const double a2 = 5.31025e-5;
    const double a1 = -1.91542e-2;
    const double a0 = 4.95369;

    return (a6*pow(lam,6) + a5*pow(lam,5) + a4*pow(lam,4) + a3*pow(lam,3) + a2*lam*lam + a1*lam + a0); // lam in nm
}

static MaterialsDB::Register<ITO> materialDB_register_ITO;

}}       // namespace plask::materials
