#include "Ni.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Ni::name() const { return NAME; }

MI_PROPERTY(Ni, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Ni, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Ni, Nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

Ni::Ni(): LorentzDrudeMetal(15.92,
                            {0.096, 0.100, 0.135, 0.106, 0.729}, // f
                            {0.048, 4.511, 1.334, 2.178, 6.292}, // G
                            {0.000, 0.174, 0.582, 1.597, 6.089}  // w
) {}

// Ni::Ni(): BrendelBormannMetal(15.92,
//                               {0.083, 0.357, 0.039, 0.127, 0.654}, // f
//                               {0.022, 2.820, 0.120, 1.822, 6.637}, // G
//                               {0.000, 0.317, 1.059, 4.583, 8.825}, // w
//                               {0.000, 0.606, 1.454, 0.379, 0.510}  // s
// ) {}


MI_PROPERTY(Ni, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: ?Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Ni::cond(double T) const {
    double tCond = 1. / (5.8e-13*pow(T-300.,2.)+4.08e-10*(T-300.)+7.19e-8);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Ni, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Ni::thermk(double T, double /*t*/) const {
    double tCondT = 90.3*pow((300./T),0.423);
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool Ni::isEqual(const Material &/*other*/) const {
    return true;
}


static MaterialsDB::Register<Ni> materialDB_register_Ni;

}}       // namespace plask::materials
