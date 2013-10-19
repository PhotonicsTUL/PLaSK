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

bool aSiO2::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<aSiO2> materialDB_register_aSiO2;

}}       // namespace plask::materials
