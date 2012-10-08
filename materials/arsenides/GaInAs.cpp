#include "GaInAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

GaInAs::GaInAs(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInAs::str() const { return StringBuilder("Ga")("In", In)("As"); }

std::string GaInAs::name() const { return NAME; }

MI_PROPERTY(GaInAs, lattC,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*mGaAs.lattC(T,'a') + In*mInAs.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*mGaAs.lattC(T,'a') + In*mInAs.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaInAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("only for Gamma point")
            )
double GaInAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaAs.Eg(T, point) + In*mInAs.Eg(T, point) - Ga*In*0.477;
    return ( tEg );
}

MI_PROPERTY(GaInAs, Dso,
            MISource("nonlinear interpolation: GaAs, InAs")
            )
double GaInAs::Dso(double T) const {
    return ( Ga*mGaAs.Dso(T) + In*mInAs.Dso(T) - Ga*In*0.15 );
}

MI_PROPERTY(GaInAs, Me,
            MISource("nonlinear interpolation: AlAs, GaAs")
            )
std::pair<double,double> GaInAs::Me(double T, char point) const {
    double lMe = Ga*mGaAs.Me(T,point).first + In*mInAs.Me(T,point).first - Ga*In*0.008,
           vMe = Ga*mGaAs.Me(T,point).second + In*mInAs.Me(T,point).second - Ga*In*0.008;
    return ( std::make_pair(lMe,vMe) );
}

MI_PROPERTY(GaInAs, Mhh,
            MISource("linear interpolation: AlAs, GaAs")
            )
std::pair<double,double> GaInAs::Mhh(double T, char point) const {
    double lMhh = Ga*mGaAs.Mhh(T,point).first + In*mInAs.Mhh(T,point).first,
           vMhh = Ga*mGaAs.Mhh(T,point).second + In*mInAs.Mhh(T,point).second;
    return ( std::make_pair(lMhh,vMhh) );
}

MI_PROPERTY(GaInAs, Mlh,
            MISource("linear interpolation: AlAs, GaAs")
            )
std::pair<double,double> GaInAs::Mlh(double T, char point) const {
    double lMlh = Ga*mGaAs.Mlh(T,point).first + In*mInAs.Mlh(T,point).first,
           vMlh = Ga*mGaAs.Mlh(T,point).second + In*mInAs.Mlh(T,point).second;
    return ( std::make_pair(lMlh,vMlh) );
}

MI_PROPERTY(GaInAs, ac,
            MISource("nonlinear interpolation: GaAs, InAs")
            )
double GaInAs::ac(double T) const {
    return ( Ga*mGaAs.ac(T) + In*mInAs.ac(T) - Ga*In*2.61 );
}

MI_PROPERTY(GaInAs, av,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::av(double T) const {
    return ( Ga*mGaAs.av(T) + In*mInAs.av(T) );
}

MI_PROPERTY(GaInAs, b,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::b(double T) const {
    return ( Ga*mGaAs.b(T) + In*mInAs.b(T) );
}

MI_PROPERTY(GaInAs, c11,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::c11(double T) const {
    return ( Ga*mGaAs.c11(T) + In*mInAs.c11(T) );
}

MI_PROPERTY(GaInAs, c12,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::c12(double T) const {
    return ( Ga*mGaAs.c12(T) + In*mInAs.c12(T) );
}

MI_PROPERTY(GaInAs, thermCond,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
std::pair<double,double> GaInAs::thermCond(double T, double t) const {
    double lCondT = 1./(Ga/mGaAs.thermCond(T,t).first + In/mInAs.thermCond(T,t).first + Ga*In*0.72),
           vCondT = 1./(Ga/mGaAs.thermCond(T,t).second + In/mInAs.thermCond(T,t).second + Ga*In*0.72);
    return ( std::make_pair(lCondT,vCondT) );
}

MI_PROPERTY(GaInAs, nr,
            MIComment("TODO")
            )
double GaInAs::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaInAs, absp,
            MIComment("TODO")
            )
double GaInAs::absp(double wl, double T) const {
    return ( 0. );
}

static MaterialsDB::Register<GaInAs> materialDB_register_GaInAs;

} // namespace plask
