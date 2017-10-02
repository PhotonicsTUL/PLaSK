#include "AlOx.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlOx::name() const { return NAME; }

MI_PROPERTY(AlOx, cond,
            MISource("A. Inoue et al., Journal of Materials Science 22 (1987) 2063-2068"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlOx::cond(double T) const {
    return ( Tensor2<double>(1e-7, 1e-7) );
}

MI_PROPERTY(AlOx, thermk,
            MISource("M. Le Du et al., Electronics Letters 42 (2006) 65-66"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlOx::thermk(double T, double h) const {
    return ( Tensor2<double>(0.7, 0.7) );
}

MI_PROPERTY(AlOx, absp,
            MISource(""),
            MIComment("TODO")
            )
double AlOx::absp(double lam, double T) const {
    return ( 0. );
}

bool AlOx::isEqual(const Material &other) const {
    return true;
}

MI_PROPERTY(AlOx, nr,
            MISource("T.Kitatani et al., Japanese Journal of Applied Physics (part1) 41 (2002) 2954-2957"),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIComment("no temperature dependence"),
            MIArgumentRange(MaterialInfo::lam, 400, 1600)
			)
double AlOx::nr(double lam, double T, double n) const {
    return ( 0.30985*exp(-lam/236.7)+1.52829 );
}

static MaterialsDB::Register<AlOx> materialDB_register_AlOx;

}}       // namespace plask::materials
