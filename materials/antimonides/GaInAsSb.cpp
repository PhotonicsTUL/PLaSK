#include "GaInAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInAsSb::GaInAsSb(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string GaInAsSb::str() const { return StringBuilder("Ga")("In", In)("As")("Sb", Sb); }

std::string GaInAsSb::name() const { return NAME; }

MI_PROPERTY(GaInAsSb, lattC,
            MISource("linear interpolation: GaSb, InSb, GaAs, InAs")
            )
double GaInAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*As*mGaAs.lattC(T,'a') + Ga*Sb*mGaSb.lattC(T,'a')
            + In*As*mInAs.lattC(T,'a') + In*Sb*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*As*mGaAs.lattC(T,'c') + Ga*Sb*mGaSb.lattC(T,'c')
            + In*As*mInAs.lattC(T,'c') + In*Sb*mInSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(GaInAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*Sb*mInSb.Eg(T,e,point)
            - Ga*In*As*(0.477) - Ga*In*Sb*(0.415) - Ga*As*Sb*(1.43) - In*As*Sb*(0.67);
    else if (point == 'X') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*Sb*mInSb.Eg(T,e,point)
            - Ga*In*As*(1.4) - Ga*In*Sb*(0.33) - Ga*As*Sb*(1.2) - In*As*Sb*(0.6);
    else if (point == 'L') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*Sb*mInSb.Eg(T,e,point)
            - Ga*In*As*(0.33) - Ga*In*Sb*(0.4) - Ga*As*Sb*(1.2) - In*As*Sb*(0.6);
    return ( tEg );
}

MI_PROPERTY(GaInAsSb, Dso,
            MISource("nonlinear interpolation: GaSb, InSb")
            )
double GaInAsSb::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*Sb*mGaSb.Dso(T,e)
             + In*As*mInAs.Dso(T,e) + In*Sb*mInSb.Dso(T,e)
             - Ga*In*As*(0.15) - Ga*In*Sb*(0.1) - Ga*As*Sb*(0.6) - In*As*Sb*(1.2) );
}

MI_PROPERTY(GaInAsSb, Me,
            MISource("nonlinear interpolation: GaSb, InSb")
            )
Tensor2<double> GaInAsSb::Me(double T, double e, char point) const {
    double lMe = Ga*As*mGaAs.Me(T,e,point).c00 + Ga*Sb*mGaSb.Me(T,e,point).c00
            + In*As*mInAs.Me(T,e,point).c00 + In*Sb*mInSb.Me(T,e,point).c00
            - Ga*In*As*(0.008) - Ga*In*Sb*(0.010) - Ga*As*Sb*(0.014) - In*As*Sb*(0.027),
           vMe = Ga*As*mGaAs.Me(T,e,point).c11 + Ga*Sb*mGaSb.Me(T,e,point).c11
            + In*As*mInAs.Me(T,e,point).c11 + In*Sb*mInSb.Me(T,e,point).c11
            - Ga*In*As*(0.008) - Ga*In*Sb*(0.010) - Ga*As*Sb*(0.014) - In*As*Sb*(0.027);
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaInAsSb, Mhh,
            MISource("linear interpolation: GaSb, InSb")
            )
Tensor2<double> GaInAsSb::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*Sb*mGaSb.Mhh(T,e).c00
            + In*As*mInAs.Mhh(T,e).c00 + In*Sb*mInSb.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*Sb*mGaSb.Mhh(T,e).c11
            + In*As*mInAs.Mhh(T,e).c11 + In*Sb*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInAsSb, Mlh,
            MISource("nonlinear interpolation: GaSb, InSb")
            )
Tensor2<double> GaInAsSb::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*Sb*mGaSb.Mlh(T,e).c00
            + In*As*mInAs.Mlh(T,e).c00 + In*Sb*mInSb.Mlh(T,e).c00 - Ga*In*Sb*(0.015),
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*Sb*mGaSb.Mlh(T,e).c11
            + In*As*mInAs.Mlh(T,e).c11 + In*Sb*mInSb.Mlh(T,e).c11 - Ga*In*Sb*(0.015);
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInAsSb, VB,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::VB(double T, double e, char point, char hole) const {
    return ( Ga*As*mGaAs.VB(T,e,point,hole) + Ga*Sb*mGaSb.VB(T,e,point,hole)
             + In*As*mInAs.VB(T,e,point,hole) + In*Sb*mInSb.VB(T,e,point,hole)
             - Ga*In*As*(-0.38) - Ga*As*Sb*(-1.06) );
}

MI_PROPERTY(GaInAsSb, ac,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*Sb*mGaSb.ac(T)
             + In*As*mInAs.ac(T) + In*Sb*mInSb.ac(T) - Ga*In*As*(2.61) );
}

MI_PROPERTY(GaInAsSb, av,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*Sb*mGaSb.av(T) + In*As*mInAs.av(T) + In*Sb*mInSb.av(T) );
}

MI_PROPERTY(GaInAsSb, b,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*Sb*mGaSb.b(T) + In*As*mInAs.b(T) + In*Sb*mInSb.b(T) );
}

MI_PROPERTY(GaInAsSb, d,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*Sb*mGaSb.d(T) + In*As*mInAs.d(T) + In*Sb*mInSb.d(T) );
}

MI_PROPERTY(GaInAsSb, c11,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*Sb*mGaSb.c11(T) + In*As*mInAs.c11(T) + In*Sb*mInSb.c11(T) );
}

MI_PROPERTY(GaInAsSb, c12,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*Sb*mGaSb.c12(T) + In*As*mInAs.c12(T) + In*Sb*mInSb.c12(T) );
}

MI_PROPERTY(GaInAsSb, c44,
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInAsSb::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*Sb*mGaSb.c44(T) + In*As*mInAs.c44(T) + In*Sb*mInSb.c44(T) );
}

MI_PROPERTY(GaInAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> GaInAsSb::thermk(double T, double t) const {
    double lCondT = 1./(Ga*As/mGaAs.thermk(T,t).c00 + Ga*Sb/mGaSb.thermk(T,t).c00
                        + In*As/mInAs.thermk(T,t).c00 + In*Sb/mInSb.thermk(T,t).c00
                        + Ga*In*0.72 + As*Sb*0.91),
           vCondT = 1./(Ga*As/mGaAs.thermk(T,t).c11 + Ga*Sb/mGaSb.thermk(T,t).c11
                        + In*As/mInAs.thermk(T,t).c11 + In*Sb/mInSb.thermk(T,t).c11
                        + Ga*In*0.72 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInAsSb, nr,
            MIComment("TODO")
            )
double GaInAsSb::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(GaInAsSb, absp,
            MIComment("TODO")
            )
double GaInAsSb::absp(double wl, double T) const {
    return ( 0. );
}

bool GaInAsSb::isEqual(const Material &other) const {
    const GaInAsSb& o = static_cast<const GaInAsSb&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInAsSb> materialDB_register_GaInAsSb;

}} // namespace plask::materials
