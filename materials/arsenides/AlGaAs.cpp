#include "AlGaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaAs::name() const { return NAME; }

std::string AlGaAs::str() const { return StringBuilder("Al", Al)("Ga")("As"); }

AlGaAs::AlGaAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(AlGaAs, thermCond,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
std::pair<double,double> AlGaAs::thermCond(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermCond(T,t).first + Ga/mGaAs.thermCond(T,t).first + Al*Ga*0.32),
           vCondT = 1./(Al/mAlAs.thermCond(T,t).second + Ga/mAlAs.thermCond(T,t).second + Al*Ga*0.32);
    return ( std::make_pair(lCondT,vCondT) );
}

MI_PROPERTY(AlGaAs, absp,
            MIComment("TODO")
            )
double AlGaAs::absp(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlGaAs, nr,
            MIComment("TODO")
            )
double AlGaAs::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlGaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815–5875"),
            MIComment("only for Gamma point")
            )
double AlGaAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T, point) + Ga*mGaAs.Eg(T, point) - Al*Ga*(-0.127+1.310*Al);
    return ( tEg );
}

MI_PROPERTY(AlGaAs, lattC,
            MISource("linear interpolation: GaAs, AlAs")
            )
double AlGaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlAs.lattC(T,'a') + Ga*mGaAs.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlAs.lattC(T,'c') + Ga*mGaAs.lattC(T,'c');
    return ( tLattC );
}

static MaterialsDB::Register<AlGaAs> materialDB_register_AlGaAs;

} // namespace plask
