#include "In.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string In::name() const { return NAME; }

MI_PROPERTY(In, cond,
            MISource("www.thinfilm.com"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> In::cond(double T) const {
    double tCond = 1.392e7;
	return (std::make_pair(tCond, tCond));
}

MI_PROPERTY(In, thermCond,
            MISource("www.lakeshore.com"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> In::thermCond(double T, double t) const {
    double tCondT = 84.;
    return(std::make_pair(tCondT, tCondT));
}

MI_PROPERTY(In, absp,
            MISource(""),
            MIComment("TODO")
            )
double In::absp(double wl, double T) const {
    return ( 1e3 );
}

MI_PROPERTY(In, nr,
            MISource(""),
            MIComment("TODO")
			)
double In::nr(double wl, double T) const {
    return ( 1. );
}

static MaterialsDB::Register<In> materialDB_register_In;

}       // namespace plask
