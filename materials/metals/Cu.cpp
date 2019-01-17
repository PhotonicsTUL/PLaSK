#include "Cu.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Cu::name() const { return NAME; }

MI_PROPERTY(Cu, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Cu, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Cu, Nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

Cu::Cu(): LorentzDrudeMetal(10.83,
                            {0.575, 0.061, 0.104, 0.723, 0.638}, // f
                            {0.030, 0.378, 1.056, 3.213, 4.305}, // G
                            {0.000, 0.291, 2.957, 5.300, 11.18}  // w
) {}

// Cu::Cu(): BrendelBormannMetal(10.83,
//                               {0.562, 0.076, 0.081, 0.324, 0.726}, // f
//                               {0.030, 0.056, 0.047, 0.113, 0.172}, // G
//                               {0.000, 0.416, 2.849, 4.819, 8.136}, // w
//                               {0.000, 0.562, 0.469, 1.131, 1.719}  // s
// ) {}

MI_PROPERTY(Cu, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: ?Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Cu::cond(double T) const {
    double tCond = 1. / (6.81e-11*(T-300.)+1.726e-8);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Cu, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Cu::thermk(double T, double /*t*/) const {
    double tCondT = 400.8*pow((300./T),0.073);
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool Cu::isEqual(const Material &/*other*/) const {
    return true;
}


static MaterialsDB::Register<Cu> materialDB_register_Cu;

}}       // namespace plask::materials
