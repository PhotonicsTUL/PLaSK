#include "AlxOy.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlxOy::name() const { return NAME; }

MI_PROPERTY(AlxOy, cond,
            MISource("A. Inoue et al., Journal of Materials Science 22 (1987) 2063-2068"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlxOy::cond(double T) const {
    return ( std::make_pair(1e-7, 1e-7) );
}

MI_PROPERTY(AlxOy, thermk,
            MISource("M. Le Du et al., Electronics Letters 42 (2006) 65-66"),
            MIComment("no temperature dependence")
            )
std::pair<double,double> AlxOy::thermk(double T, double t) const {
    return ( std::make_pair(0.7, 0.7) );
}

MI_PROPERTY(AlxOy, absp,
            MISource(""),
            MIComment("TODO")
            )
double AlxOy::absp(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlxOy, nr,
            MISource("T.Kitatani et al., Japanese Journal of Applied Physics (part1) 41 (2002) 2954-2957"),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIComment("no temperature dependence"),
            MIArgumentRange(MaterialInfo::wl, 400, 1600)
			)
double AlxOy::nr(double wl, double T) const {
    return ( 0.30985*exp(-wl/236.7)+1.52829 );
}

static MaterialsDB::Register<AlxOy> materialDB_register_AlxOy;

}       // namespace plask
