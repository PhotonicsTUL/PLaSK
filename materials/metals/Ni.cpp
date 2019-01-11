#include "Ni.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Ni::name() const { return NAME; }

MI_PROPERTY(Ni, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MIComment("fit from: ?Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Ni::cond(double T) const {
    double tCond = 1. / (5.8e-13*pow(T-300.,2.)+4.08e-10*(T-300.)+7.19e-8);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Ni, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MIComment("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Ni::thermk(double T, double /*t*/) const {
    double tCondT = 90.3*pow((300./T),0.423);
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool Ni::isEqual(const Material &/*other*/) const {
    return true;
}

 MI_PROPERTY(Ni, absp,
             MISource(""),
             MIComment("TODO")
             )
 double Ni::absp(double lam, double T) const {
	 return optpar("LD", "abs", name(), lam);
 }

 MI_PROPERTY(Ni, nr,
			 MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
             MIComment("no temperature dependence")
 			)
 double Ni::nr(double lam, double T, double n) const {
	 return optpar("LD", "nr", name(), lam);
 }

static MaterialsDB::Register<Ni> materialDB_register_Ni;

}}       // namespace plask::materials
