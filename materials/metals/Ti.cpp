#include "Ti.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Ti::name() const { return NAME; }

MI_PROPERTY(Ti, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Ti, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

Ti::Ti(): LorentzDrudeMetal(7.29,
                            {0.148, 0.899, 0.393, 0.187, 0.001}, // f
                            {0.082, 2.276, 2.518, 1.663, 1.762}, // G
                            {0.000, 0.777, 1.545, 2.509, 19.43}  // w
) {}

// Ti::Ti(): BrendelBormannMetal(7.29,
//                               {0.126, 0.427, 0.218, 0.513, 0.0002},// f
//                               {0.067, 1.877, 0.100, 0.615, 4.109}, // G
//                               {0.000, 1.459, 2.661, 0.805, 19.86}, // w
//                               {0.000, 0.463, 0.506, 0.799, 2.854}  // s
// ) {}


MI_PROPERTY(Ti, cond,
            MISource("N.D. Milosevic and K.D. Maglic, Thermophysical properties of solid phase titanium in a wide temperature range, High Temperatures-High Pressures, vol. 37, pp. 187-204, 2008."),
            MIArgumentRange(MaterialInfo::T, 250, 1100)
            )
Tensor2<double> Ti::cond(double T) const {
    double tCond = 1. / (6.17169e-8 + 9.010579e-10*T + 1.817669e-12*T*T - 1.225226e-15*T*T*T);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Ti, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Ti::thermk(double T, double /*t*/) const {
    double tCondT = 22.00*pow((300./T),0.24);
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool Ti::isEqual(const Material &/*other*/) const {
    return true;
}

//MI_PROPERTY(Ti, absp,
//            MISource(""),
//            MIComment("TODO"),
//            MIArgumentRange(MaterialInfo::lam, 480, 20700)
//            )
//double Ti::absp(double lam, double /*T*/) const {
//    double ulam = lam*1e-3;
//    if (ulam<1000.)
//        return ( (4.75779 -19.2528*ulam + 34.0917*ulam*ulam -27.2725*pow(ulam,3.) + 8.1585*pow(ulam,4.))*1e6 );
//    else
//        return ( 864255*pow(exp(ulam),-1.18177) + 209715 + 6708.34*(ulam) - 633.799*ulam*ulam + 12.9902*pow(ulam,3.) );
//}
//
//MI_PROPERTY(Ti, nr,
//            MISource(""),
//            MIComment("TODO"),
//            MIArgumentRange(MaterialInfo::lam, 480, 20700)
//            )
//double Ti::nr(double lam, double /*T*/, double /*n*/) const {
//    double ulam = lam*1e-3;
//    return ( -0.443425 + 5.15294*ulam - 2.15683*ulam*ulam + 0.466666*pow(ulam,3.) - 0.0571905*pow(ulam,4.) + 0.00423617*pow(ulam,5.) - 0.000187612*pow(ulam,6.) + 4.56964e-6*pow(ulam,7.) - 4.70605e-8*pow(ulam,8.) );
//}

static MaterialsDB::Register<Ti> materialDB_register_Ti;

}}       // namespace plask::materials
