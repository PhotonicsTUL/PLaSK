#include "GaInSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInSb::GaInSb(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInSb::str() const { return StringBuilder("Ga")("In", In)("Sb"); }

std::string GaInSb::name() const { return NAME; }

MI_PROPERTY(GaInSb, lattC,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*mGaSb.lattC(T,'a') + In*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*mGaSb.lattC(T,'c') + In*mInSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(GaInSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Ga*In*(0.415);
    else if (point == 'X') tEg = Ga*mGaSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Ga*In*(0.33);
    else if (point == 'L') tEg = Ga*mGaSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Ga*In*(0.4);
    return ( tEg );
}

MI_PROPERTY(GaInSb, Dso,
            MISource("nonlinear interpolation: GaSb, InSb")
            )
double GaInSb::Dso(double T, double e) const {
    return ( Ga*mGaSb.Dso(T,e) + In*mInSb.Dso(T,e) - Ga*In*0.1 );
}

MI_PROPERTY(GaInSb, Me,
            MISource("nonlinear interpolation: GaSb, InSb")
            )
Tensor2<double> GaInSb::Me(double T, double e, char point) const {
    double lMe = Ga*mGaSb.Me(T,e,point).c00 + In*mInSb.Me(T,e,point).c00 - Ga*In*0.010,
           vMe = Ga*mGaSb.Me(T,e,point).c11 + In*mInSb.Me(T,e,point).c11 - Ga*In*0.010;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaInSb, Mhh,
            MISource("linear interpolation: GaSb, InSb")
            )
Tensor2<double> GaInSb::Mhh(double T, double e) const {
    double lMhh = Ga*mGaSb.Mhh(T,e).c00 + In*mInSb.Mhh(T,e).c00,
           vMhh = Ga*mGaSb.Mhh(T,e).c11 + In*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInSb, Mlh,
            MISource("nonlinear interpolation: GaSb, InSb")
            )
Tensor2<double> GaInSb::Mlh(double T, double e) const {
    double lMlh = Ga*mGaSb.Mlh(T,e).c00 + In*mInSb.Mlh(T,e).c00 - Ga*In*0.015,
           vMlh = Ga*mGaSb.Mlh(T,e).c11 + In*mInSb.Mlh(T,e).c11 - Ga*In*0.015;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInSb, VB,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*mGaSb.VB(T,e,point,hole) + In*mInSb.VB(T,e,point,hole) );
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
    }
    return tVB;
}

MI_PROPERTY(GaInSb, ac,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::ac(double T) const {
    return ( Ga*mGaSb.ac(T) + In*mInSb.ac(T) );
}

MI_PROPERTY(GaInSb, av,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::av(double T) const {
    return ( Ga*mGaSb.av(T) + In*mInSb.av(T) );
}

MI_PROPERTY(GaInSb, b,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::b(double T) const {
    return ( Ga*mGaSb.b(T) + In*mInSb.b(T) );
}

MI_PROPERTY(GaInSb, d,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::d(double T) const {
    return ( Ga*mGaSb.d(T) + In*mInSb.d(T) );
}

MI_PROPERTY(GaInSb, c11,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::c11(double T) const {
    return ( Ga*mGaSb.c11(T) + In*mInSb.c11(T) );
}

MI_PROPERTY(GaInSb, c12,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::c12(double T) const {
    return ( Ga*mGaSb.c12(T) + In*mInSb.c12(T) );
}

MI_PROPERTY(GaInSb, c44,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::c44(double T) const {
    return ( Ga*mGaSb.c44(T) + In*mInSb.c44(T) );
}

MI_PROPERTY(GaInSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> GaInSb::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaSb.thermk(T,t).c00 + In/mInSb.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaSb.thermk(T,t).c11 + In/mGaSb.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInSb, nr,
            MIComment("TODO")
            )
double GaInSb::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(GaInSb, absp,
            MIComment("TODO")
            )
double GaInSb::absp(double wl, double T) const {
    return ( 0. );
}

bool GaInSb::isEqual(const Material &other) const {
    const GaInSb& o = static_cast<const GaInSb&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInSb> materialDB_register_GaInSb;

}} // namespace plask::materials
