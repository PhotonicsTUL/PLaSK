#include "Al.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Al::name() const { return NAME; }


MI_PROPERTY(Al, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)


MI_PROPERTY(Al, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Al, Nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

Al::Al(): LorentzDrudeMetal(14.98,
                            {0.523, 0.227, 0.050, 0.166, 0.030}, // f
                            {0.047, 0.333, 0.312, 1.351, 3.382}, // G
                            {0.000, 0.162, 1.544, 1.808, 3.473}  // w
) {}

// Al::Al(): BrendelBormannMetal(14.98,
//                               {0.526, 0.213, 0.060, 0.182, 0.014}, // f
//                               {0.047, 0.312, 0.315, 1.587, 2.145}, // G
//                               {0.000, 0.163, 1.561, 1.827, 4.495}, // w
//                               {0.000, 0.013, 0.042, 0.256, 1.735}  // s
// ) {}


bool Al::isEqual(const Material &/*other*/) const {
    return true;
}


static MaterialsDB::Register<Al> materialDB_register_Al;

}} // namespace plask::materials
