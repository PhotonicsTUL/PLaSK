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

bool aSi::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSi> materialDB_register_aSi;

}}       // namespace plask::materials
