#include "In.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string In::name() const { return NAME; }

MI_PROPERTY(In, cond,
            MISource("www.thinfilm.com"),
            MIComment("no temperature dependence")
            )
Tensor2<double> In::cond(double T) const {
    double tCond = 1.392e7;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(In, thermk,
            MISource("www.lakeshore.com"),
            MIComment("no temperature dependence")
            )
Tensor2<double> In::thermk(double T, double t) const {
    double tCondT = 84.;
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(In, absp,
            MISource(""),
            MIComment("TODO")
            )
double In::absp(double lam, double T) const {
    return ( 1e3 );
}

bool In::isEqual(const Material &other) const {
    return true;
}

MI_PROPERTY(In, nr,
            MISource(""),
            MIComment("TODO")
			)
double In::nr(double lam, double T, double n) const {
    return ( 1. );
}

static MaterialsDB::Register<In> materialDB_register_In;

}}       // namespace plask::materials
