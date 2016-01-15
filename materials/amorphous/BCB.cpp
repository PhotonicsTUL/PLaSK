#include "BCB.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string BCB::name() const { return NAME; }

MI_PROPERTY(BCB, cond,
            MISource("The DOW Chemical Company, CYCLOTENE Advanced Electronic Resins (2005) 1-9"),
            MIComment("no temperature dependence")
            )
Tensor2<double> BCB::cond(double T) const {
    return ( Tensor2<double>(1e-17, 1e-17) );
}

MI_PROPERTY(BCB, thermk,
            MISource("X. Xu et al., IEEE Components, Packaging, and Manufacturing Technology 2 (2012) 286-293"),
            MIComment("fit for pure BCB by Lukasz Piskorski, unpublished"),
            MIArgumentRange(MaterialInfo::T, 290, 420)
            )
Tensor2<double> BCB::thermk(double T, double h) const {
    double tK = 0.31*pow(300./T,-1.1); // [tK] = W/(m*K)
    return ( Tensor2<double>(tK, tK) );
}

Material::ConductivityType BCB::condtype() const { return Material::CONDUCTIVITY_OTHER; }

MI_PROPERTY(BCB, dens,
            MISource("A. Modafe et al., Microelectronic Engineering 82 (2005) 154-167")
            )
double BCB::dens(double T) const {
    return ( 1050. ); // kg/m^3
}

MI_PROPERTY(BCB, cp,
            MISource("A. Modafe et al., Microelectronic Engineering 82 (2005) 154-167")
            )
double BCB::cp(double T) const {
    return ( 2180. ); // J/(kg*K)
}

bool BCB::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<BCB> materialDB_register_BCB;

}}       // namespace plask::materials
