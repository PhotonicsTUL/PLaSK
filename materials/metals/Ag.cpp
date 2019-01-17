#include "Ag.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Ag::name() const { return NAME; }


MI_PROPERTY(Ag, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Ag, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

MI_PROPERTY(Ag, Nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MIComment("no temperature dependence")
)

Ag::Ag(): LorentzDrudeMetal(9.01,
                            {0.845, 0.065, 0.124, 0.011, 0.840, 5.646}, // f
                            {0.048, 3.886, 0.452, 0.065, 0.916, 2.419}, // G
                            {0.000, 0.816, 4.481, 8.185, 9.083, 20.29}  // w
) {}

// Ag::Ag(): BrendelBormannMetal(9.01,
//                               {0.821, 0.050, 0.133, 0.051, 0.467, 4.000}, // f
//                               {0.049, 0.189, 0.067, 0.019, 0.117, 0.052}, // G
//                               {0.000, 2.025, 5.185, 4.343, 9.809, 18.56}, // w
//                               {0.000, 1.894, 0.665, 0.189, 1.170, 0.516}  // s
// ) {}


bool Ag::isEqual(const Material &/*other*/) const {
    return true;
}


static MaterialsDB::Register<Ag> materialDB_register_Ag;

}} // namespace plask::materials
