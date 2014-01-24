#include "AlInSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlInSb::AlInSb(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    In = Comp.find("In")->second;
}

std::string AlInSb::str() const { return StringBuilder("Al", Al)("In")("Sb"); }

std::string AlInSb::name() const { return NAME; }

MI_PROPERTY(AlInSb, lattC,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlSb.lattC(T,'a') + In*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlSb.lattC(T,'c') + In*mInSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlInSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlInSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Al*In*(0.43);
    else if (point == 'X') tEg = Al*mAlSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point);
    else if (point == 'L') tEg = Al*mAlSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point);
    return ( tEg );
}

MI_PROPERTY(AlInSb, Dso,
            MISource("nonlinear interpolation: AlSb, InSb")
            )
double AlInSb::Dso(double T, double e) const {
    return ( Al*mAlSb.Dso(T,e) + In*mInSb.Dso(T,e) - Al*In*0.25 );
}

MI_PROPERTY(AlInSb, Me,
            MISource("linear interpolation: AlSb, InSb")
            )
Tensor2<double> AlInSb::Me(double T, double e, char point) const {
    double lMe = Al*mAlSb.Me(T,e,point).c00 + In*mInSb.Me(T,e,point).c00,
           vMe = Al*mAlSb.Me(T,e,point).c11 + In*mInSb.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlInSb, Mhh,
            MISource("linear interpolation: AlSb, InSb")
            )
Tensor2<double> AlInSb::Mhh(double T, double e) const {
    double lMhh = Al*mAlSb.Mhh(T,e).c00 + In*mInSb.Mhh(T,e).c00,
           vMhh = Al*mAlSb.Mhh(T,e).c11 + In*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlInSb, Mlh,
            MISource("linear interpolation: AlSb, InSb")
            )
Tensor2<double> AlInSb::Mlh(double T, double e) const {
    double lMlh = Al*mAlSb.Mlh(T,e).c00 + In*mInSb.Mlh(T,e).c00,
           vMlh = Al*mAlSb.Mlh(T,e).c11 + In*mInSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlInSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlInSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlInSb, VB,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlSb.VB(T,e,point,hole) + In*mInSb.VB(T,e,point,hole) );
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
    }
    return tVB;
}

MI_PROPERTY(AlInSb, ac,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::ac(double T) const {
    return ( Al*mAlSb.ac(T) + In*mInSb.ac(T) );
}

MI_PROPERTY(AlInSb, av,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::av(double T) const {
    return ( Al*mAlSb.av(T) + In*mInSb.av(T) );
}

MI_PROPERTY(AlInSb, b,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::b(double T) const {
    return ( Al*mAlSb.b(T) + In*mInSb.b(T) );
}

MI_PROPERTY(AlInSb, d,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::d(double T) const {
    return ( Al*mAlSb.d(T) + In*mInSb.d(T) );
}

MI_PROPERTY(AlInSb, c11,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::c11(double T) const {
    return ( Al*mAlSb.c11(T) + In*mInSb.c11(T) );
}

MI_PROPERTY(AlInSb, c12,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::c12(double T) const {
    return ( Al*mAlSb.c12(T) + In*mInSb.c12(T) );
}

MI_PROPERTY(AlInSb, c44,
            MISource("linear interpolation: AlSb, InSb")
            )
double AlInSb::c44(double T) const {
    return ( Al*mAlSb.c44(T) + In*mInSb.c44(T) );
}

MI_PROPERTY(AlInSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> AlInSb::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlSb.thermk(T,t).c00 + In/mInSb.thermk(T,t).c00 + Al*In*0.15),
           vCondT = 1./(Al/mAlSb.thermk(T,t).c11 + In/mAlSb.thermk(T,t).c11 + Al*In*0.15);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlInSb, nr,
            MIComment("TODO")
            )
double AlInSb::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(AlInSb, absp,
            MIComment("TODO")
            )
double AlInSb::absp(double wl, double T) const {
    return ( 0. );
}

bool AlInSb::isEqual(const Material &other) const {
    const AlInSb& o = static_cast<const AlInSb&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlInSb> materialDB_register_AlInSb;

}} // namespace plask::materials
