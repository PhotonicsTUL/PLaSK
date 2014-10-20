#include "AlGaSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaSb::AlGaSb(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

std::string AlGaSb::str() const { return StringBuilder("Al", Al)("Ga")("Sb"); }

std::string AlGaSb::name() const { return NAME; }

MI_PROPERTY(AlGaSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlSb.lattC(T,'a') + Ga*mGaSb.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlSb.lattC(T,'c') + Ga*mGaSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlGaSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: AlSb, GaSb")
            )
double AlGaSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlSb.Eg(T,e,point) + Ga*mGaSb.Eg(T,e,point) - Al*Ga*(-0.044+1.22*Al);
    else if (point == 'X') tEg = Al*mAlSb.Eg(T,e,point) + Ga*mGaSb.Eg(T,e,point);
    else if (point == 'L') tEg = Al*mAlSb.Eg(T,e,point) + Ga*mGaSb.Eg(T,e,point);
    else if (point == '*')
    {
        double tEgG = Al*mAlSb.Eg(T,e,'G') + Ga*mGaSb.Eg(T,e,'G') - Al*Ga*(-0.044+1.22*Al);
        double tEgX = Al*mAlSb.Eg(T,e,'X') + Ga*mGaSb.Eg(T,e,'X');
        double tEgL = Al*mAlSb.Eg(T,e,'L') + Ga*mGaSb.Eg(T,e,'L');
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: AlSb, GaSb")
            )
double AlGaSb::Dso(double T, double e) const {
    return ( Al*mAlSb.Dso(T,e) + Ga*mGaSb.Dso(T,e) - Al*Ga*0.3 );
}

MI_PROPERTY(AlGaSb, Me,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlSb, GaSb")
            )
Tensor2<double> AlGaSb::Me(double T, double e, char point) const {
    double lMe = Al*mAlSb.Me(T,e,point).c00 + Ga*mGaSb.Me(T,e,point).c00,
           vMe = Al*mAlSb.Me(T,e,point).c11 + Ga*mGaSb.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlGaSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlSb, GaSb")
            )
Tensor2<double> AlGaSb::Mhh(double T, double e) const {
    double lMhh = Al*mAlSb.Mhh(T,e).c00 + Ga*mGaSb.Mhh(T,e).c00,
           vMhh = Al*mAlSb.Mhh(T,e).c11 + Ga*mGaSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlSb, GaSb")
            )
Tensor2<double> AlGaSb::Mlh(double T, double e) const {
    double lMlh = Al*mAlSb.Mlh(T,e).c00 + Ga*mGaSb.Mlh(T,e).c00,
           vMlh = Al*mAlSb.Mlh(T,e).c11 + Ga*mGaSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlSb.VB(T,0.,point,hole) + Ga*mGaSb.VB(T,0.,point,hole) );
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
    }
    return tVB;
}

MI_PROPERTY(AlGaSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::ac(double T) const {
    return ( Al*mAlSb.ac(T) + Ga*mGaSb.ac(T) );
}

MI_PROPERTY(AlGaSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::av(double T) const {
    return ( Al*mAlSb.av(T) + Ga*mGaSb.av(T) );
}

MI_PROPERTY(AlGaSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::b(double T) const {
    return ( Al*mAlSb.b(T) + Ga*mGaSb.b(T) );
}

MI_PROPERTY(AlGaSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::d(double T) const {
    return ( Al*mAlSb.d(T) + Ga*mGaSb.d(T) );
}

MI_PROPERTY(AlGaSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::c11(double T) const {
    return ( Al*mAlSb.c11(T) + Ga*mGaSb.c11(T) );
}

MI_PROPERTY(AlGaSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::c12(double T) const {
    return ( Al*mAlSb.c12(T) + Ga*mGaSb.c12(T) );
}

MI_PROPERTY(AlGaSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlSb, GaSb")
            )
double AlGaSb::c44(double T) const {
    return ( Al*mAlSb.c44(T) + Ga*mGaSb.c44(T) );
}

MI_PROPERTY(AlGaSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MISource("inversion od nonlinear interpolation of resistivity: AlSb, GaSb")
            )
Tensor2<double> AlGaSb::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlSb.thermk(T,t).c00 + Ga/mGaSb.thermk(T,t).c00 + Al*Ga*0.32),
           vCondT = 1./(Al/mAlSb.thermk(T,t).c11 + Ga/mAlSb.thermk(T,t).c11 + Al*Ga*0.32);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlSb, GaSb"),
            MIComment("no temperature dependence")
            )
double AlGaSb::dens(double T) const {
    return ( Al*mAlSb.dens(T) + Ga*mGaSb.dens(T) );
}

MI_PROPERTY(AlGaSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlSb, GaSb"),
            MIComment("no temperature dependence")
            )
double AlGaSb::cp(double T) const {
    return ( Al*mAlSb.cp(T) + Ga*mGaSb.cp(T) );
}

MI_PROPERTY(AlGaSb, nr,
            MIComment("TODO")
            )
double AlGaSb::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(AlGaSb, absp,
            MIComment("TODO")
            )
double AlGaSb::absp(double wl, double T) const {
    return ( 0. );
}

bool AlGaSb::isEqual(const Material &other) const {
    const AlGaSb& o = static_cast<const AlGaSb&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaSb> materialDB_register_AlGaSb;

}} // namespace plask::materials
