#include "GaInAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInAs::GaInAs(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInAs::str() const { return StringBuilder("In", In)("Ga")("As"); }

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
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Ga*In*0.477;
    else if (point == 'X') tEg = Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Ga*In*1.4;
    else if (point == 'L') tEg = Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Ga*In*0.33;
    return ( tEg );
}

MI_PROPERTY(GaInAs, Dso,
            MISource("nonlinear interpolation: GaAs, InAs")
            )
double GaInAs::Dso(double T, double e) const {
    return ( Ga*mGaAs.Dso(T, e) + In*mInAs.Dso(T, e) - Ga*In*0.15 );
}

MI_PROPERTY(GaInAs, Me,
            MISource("nonlinear interpolation: AlAs, GaAs")
            )
Tensor2<double> GaInAs::Me(double T, double e, char point) const {
    double lMe = Ga*mGaAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00 - Ga*In*0.008,
           vMe = Ga*mGaAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11 - Ga*In*0.008;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaInAs, Mhh,
            MISource("linear interpolation: AlAs, GaAs")
            )
Tensor2<double> GaInAs::Mhh(double T, double e) const {
    double lMhh = Ga*mGaAs.Mhh(T,e).c00 + In*mInAs.Mhh(T,e).c00,
           vMhh = Ga*mGaAs.Mhh(T,e).c11 + In*mInAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInAs, Mlh,
            MISource("linear interpolation: AlAs, GaAs")
            )
Tensor2<double> GaInAs::Mlh(double T, double e) const {
    double lMlh = Ga*mGaAs.Mlh(T,e).c00 + In*mInAs.Mlh(T,e).c00,
           vMlh = Ga*mGaAs.Mlh(T,e).c11 + In*mInAs.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInAs, VB,
            MISource("nonlinear interpolation: GaAs, InAs")
            )
double GaInAs::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*mGaAs.VB(T,e,point,hole) + In*mInAs.VB(T,e,point,hole) - Ga*In*(-0.38) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
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

MI_PROPERTY(GaInAs, d,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::d(double T) const {
    return ( Ga*mGaAs.d(T) + In*mInAs.d(T) );
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

MI_PROPERTY(GaInAs, c44,
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::c44(double T) const {
    return ( Ga*mGaAs.c44(T) + In*mInAs.c44(T) );
}

MI_PROPERTY(GaInAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> GaInAs::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaAs.thermk(T,t).c00 + In/mInAs.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaAs.thermk(T,t).c11 + In/mInAs.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInAs, nr,
            MIComment("TODO")
            )
double GaInAs::nr(double wl, double T, double n) const {
    throw NotImplemented("nR for GaInAs");
    //return ( 0. );
}

MI_PROPERTY(GaInAs, absp,
            MIComment("TODO")
            )
double GaInAs::absp(double wl, double T) const {
    throw NotImplemented("absp for GaInAs");
    //return ( 0. );
}

double GaInAs::eps(double T) const {
    return In*mInAs.eps(T) + Ga*mGaAs.eps(T);
}

bool GaInAs::isEqual(const Material &other) const {
    const GaInAs& o = static_cast<const GaInAs&>(other);
    return o.Ga == this->Ga;
}

static MaterialsDB::Register<GaInAs> materialDB_register_GaInAs;

}} // namespace plask::materials
