#include "AlGaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

AlGaAs::AlGaAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

std::string AlGaAs::str() const { return StringBuilder("Al", Al)("Ga")("As"); }

std::string AlGaAs::name() const { return NAME; }

MI_PROPERTY(AlGaAs, lattC,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlAs.lattC(T,'a') + Ga*mGaAs.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlAs.lattC(T,'c') + Ga*mGaAs.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlGaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double AlGaAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T, point) + Ga*mGaAs.Eg(T, point) - Al*Ga*(-0.127+1.310*Al);
    return ( tEg );
}

MI_PROPERTY(AlGaAs, Dso,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::Dso(double T) const {
    return ( Al*mAlAs.Dso(T) + Ga*mGaAs.Dso(T) );
}

MI_PROPERTY(AlGaAs, Me,
            MISource("linear interpolation: AlAs, GaAs")
            )
std::pair<double,double> AlGaAs::Me(double T, char point) const {
    double lMe = Al*mAlAs.Me(T,point).first + Ga*mGaAs.Me(T,point).first,
           vMe = Al*mAlAs.Me(T,point).second + Ga*mGaAs.Me(T,point).second;
    return ( std::make_pair(lMe,vMe) );
}

MI_PROPERTY(AlGaAs, Mhh,
            MISource("linear interpolation: AlAs, GaAs")
            )
std::pair<double,double> AlGaAs::Mhh(double T, char point) const {
    double lMhh = Al*mAlAs.Mhh(T,point).first + Ga*mGaAs.Mhh(T,point).first,
           vMhh = Al*mAlAs.Mhh(T,point).second + Ga*mGaAs.Mhh(T,point).second;
    return ( std::make_pair(lMhh,vMhh) );
}

MI_PROPERTY(AlGaAs, Mlh,
            MISource("linear interpolation: AlAs, GaAs")
            )
std::pair<double,double> AlGaAs::Mlh(double T, char point) const {
    double lMlh = Al*mAlAs.Mlh(T,point).first + Ga*mGaAs.Mlh(T,point).first,
           vMlh = Al*mAlAs.Mlh(T,point).second + Ga*mGaAs.Mlh(T,point).second;
    return ( std::make_pair(lMlh,vMlh) );
}

MI_PROPERTY(AlGaAs, ac,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::ac(double T) const {
    return ( Al*mAlAs.ac(T) + Ga*mGaAs.ac(T) );
}

MI_PROPERTY(AlGaAs, av,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::av(double T) const {
    return ( Al*mAlAs.av(T) + Ga*mGaAs.av(T) );
}

MI_PROPERTY(AlGaAs, b,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::b(double T) const {
    return ( Al*mAlAs.b(T) + Ga*mGaAs.b(T) );
}

MI_PROPERTY(AlGaAs, c11,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::c11(double T) const {
    return ( Al*mAlAs.c11(T) + Ga*mGaAs.c11(T) );
}

MI_PROPERTY(AlGaAs, c12,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::c12(double T) const {
    return ( Al*mAlAs.c12(T) + Ga*mGaAs.c12(T) );
}

MI_PROPERTY(AlGaAs, thermCond,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
std::pair<double,double> AlGaAs::thermCond(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermCond(T,t).first + Ga/mGaAs.thermCond(T,t).first + Al*Ga*0.32),
           vCondT = 1./(Al/mAlAs.thermCond(T,t).second + Ga/mAlAs.thermCond(T,t).second + Al*Ga*0.32);
    return ( std::make_pair(lCondT,vCondT) );
}

MI_PROPERTY(AlGaAs, nr,
            MIComment("TODO")
            )
double AlGaAs::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlGaAs, absp,
            MIComment("TODO")
            )
double AlGaAs::absp(double wl, double T) const {
    return ( 0. );
}

static MaterialsDB::Register<AlGaAs> materialDB_register_AlGaAs;

} // namespace plask
